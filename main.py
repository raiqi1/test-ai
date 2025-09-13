# main.py - Enhanced FastAPI Backend with HERE Maps Integration
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import re
import requests
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import logging
from geopy.distance import geodesic
import time
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        initialize_clients()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")

app = FastAPI(
    title="LLM Location Assistant with HERE Maps",
    description="AI-powered location search with HERE Maps integration for comprehensive coverage",
    version="3.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory if it exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class LocationQuery(BaseModel):
    query: str
    location: Optional[str] = "Jakarta, Indonesia"
    radius: Optional[int] = 5000
    lat: Optional[float] = None
    lng: Optional[float] = None

class LocationResult(BaseModel):
    name: str
    address: str
    rating: Optional[float] = None
    lat: float
    lng: float
    distance: Optional[float] = None
    category: str
    place_id: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    here_url: str

class ChatResponse(BaseModel):
    response: str
    locations: List[LocationResult]
    map_center: Dict[str, float]
    search_query: str
    total_found: int

# Global variables
llm_client = None
HERE_API_KEY = "tFlpJjB2Q5vD4F2fm5hN8fma9gQ-dOarA6XZy4iSMkk"

def initialize_clients():
    """Initialize OpenRouter client"""
    global llm_client
    
    try:
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            
        llm_client = ChatOpenAI(
            model=os.getenv("MODEL", "deepseek/deepseek-chat-v3"),
            api_key=openrouter_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=0.7,
            max_tokens=1000
        )
        logger.info("OpenRouter client initialized")
        
        # Test connection
        test_response = llm_client.invoke("Hello, test connection")
        logger.info("OpenRouter connection tested successfully")
        
        # Validate HERE API key
        if not HERE_API_KEY:
            logger.warning("HERE_API_KEY not provided - some features may be limited")
        else:
            logger.info("HERE Maps API key configured")
        
    except Exception as e:
        logger.error(f"Error initializing clients: {str(e)}")
        raise

def get_llm_client():
    if llm_client is None:
        initialize_clients()
    return llm_client

def extract_search_intent(query: str, llm: ChatOpenAI) -> Dict[str, Any]:
    """Enhanced LLM intent extraction with HERE Maps categories"""
    
    system_prompt = """You are a location search assistant. Given a user query, extract:
1. The specific brand/name they're looking for
2. The type of place they're looking for  
3. Any specific preferences
4. Location context if mentioned

Respond in JSON format:
{
    "search_type": "restaurant|cafe|hospital|shop|tourism|entertainment|transport|sport|etc",
    "keywords": "specific search terms for search",
    "brand_name": "specific brand name if mentioned",
    "location_context": "specific area if mentioned",
    "preferences": ["any specific preferences"],
    "here_categories": ["list of HERE Maps POI categories"],
    "is_brand_specific": true/false
}

HERE MAPS CATEGORY MAPPING:
- "tempat makan/restaurant" -> ["restaurant", "food-drink", "eat-drink"]
- "kolam renang" -> ["leisure-outdoor", "sports-recreation", "swimming-pool"]
- "tempat olahraga/gym" -> ["sports-recreation", "fitness", "gym"]
- "tempat wisata" -> ["sights-museums", "tourist-attraction", "landmark-attraction"]
- "hotel" -> ["accommodation", "hotel"]
- "rumah sakit" -> ["hospital-health-care-facility", "medical-services"]
- "sekolah" -> ["education-facility", "school"]
- "mall/shopping" -> ["shopping", "department-store", "shopping-mall"]
- "cafe" -> ["coffee-tea", "restaurant", "eat-drink"]
- "bank/ATM" -> ["atm-bank-exchange", "bank"]
- "gas station" -> ["petrol-station"]

Examples:
- "mie gacoan" -> {"search_type": "restaurant", "keywords": "Mie Gacoan", "brand_name": "Mie Gacoan", "is_brand_specific": true, "here_categories": ["restaurant", "eat-drink"]}
- "tempat makan" -> {"search_type": "restaurant", "keywords": "restaurant food", "here_categories": ["restaurant", "eat-drink"], "is_brand_specific": false}
- "kolam renang" -> {"search_type": "sport", "keywords": "swimming pool", "here_categories": ["sports-recreation", "swimming-pool"], "is_brand_specific": false}
"""
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}")
        ]
        
        response = llm_client.invoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # Set default values
            result.setdefault("is_brand_specific", False)
            result.setdefault("brand_name", None)
            result.setdefault("preferences", [])
            result.setdefault("location_context", None)
            result.setdefault("here_categories", ["restaurant"])
            return result
        else:
            return {
                "search_type": "restaurant",
                "keywords": query,
                "brand_name": None,
                "location_context": None,
                "preferences": [],
                "here_categories": ["restaurant"],
                "is_brand_specific": False
            }
    
    except Exception as e:
        logger.error(f"Error in extract_search_intent: {str(e)}")
        return {
            "search_type": "restaurant", 
            "keywords": query,
            "brand_name": None,
            "location_context": None,
            "preferences": [],
            "here_categories": ["restaurant"],
            "is_brand_specific": False
        }

def search_with_here_geocoding(search_intent: Dict[str, Any], lat: float, lon: float, radius_km: float) -> List[Dict]:
    """Search using HERE Geocoding API"""
    
    try:
        keywords = search_intent.get("keywords", "")
        
        # HERE Geocoding API endpoint
        geocoding_url = "https://geocode.search.hereapi.com/v1/geocode"
        
        params = {
            "q": keywords,
            "at": f"{lat},{lon}",
            "limit": 20,
            "lang": "id-ID",
            "apiKey": HERE_API_KEY
        }
        
        logger.info(f"HERE Geocoding API request: {geocoding_url}")
        logger.info(f"Params: {params}")
        
        response = requests.get(geocoding_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            
            results = []
            for item in items:
                position = item.get('position', {})
                address = item.get('address', {})
                
                if position.get('lat') and position.get('lng'):
                    result_lat, result_lng = position['lat'], position['lng']
                    
                    # Check distance
                    distance = geodesic((lat, lon), (result_lat, result_lng)).kilometers
                    if distance <= radius_km:
                        # Extract place info
                        title = item.get('title', 'Unknown Place')
                        label = address.get('label', title)
                        
                        results.append({
                            'lat': result_lat,
                            'lon': result_lng,
                            'name': title,
                            'address': label,
                            'category': 'place',
                            'here_id': item.get('id', ''),
                            'distance_km': distance,
                            'score': item.get('scoring', {}).get('queryScore', 0)
                        })
            
            logger.info(f"HERE Geocoding found {len(results)} results")
            return results
        else:
            logger.error(f"HERE Geocoding API error: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        logger.error(f"Error in search_with_here_geocoding: {str(e)}")
        return []

def search_with_here_places(search_intent: Dict[str, Any], lat: float, lon: float, radius_m: int) -> List[Dict]:
    """Search using HERE Places API"""
    
    try:
        keywords = search_intent.get("keywords", "")
        here_categories = search_intent.get("here_categories", [])
        is_brand_specific = search_intent.get("is_brand_specific", False)
        
        # HERE Places API endpoint for nearby search
        places_url = "https://discover.search.hereapi.com/v1/discover"
        
        params = {
            "at": f"{lat},{lon}",
            "limit": 25,
            "lang": "id-ID",
            "apiKey": HERE_API_KEY
        }
        
        # Add query if available
        if keywords:
            params["q"] = keywords
        
        # Add category filter if not brand-specific
        if not is_brand_specific and here_categories:
            # Map our categories to HERE categories
            here_cat_mapping = {
                "restaurant": "600-6000-0000",
                "eat-drink": "600-6000-0000",
                "food-drink": "600-6000-0000",
                "coffee-tea": "600-6300-0000",
                "accommodation": "500-5000-0000",
                "hotel": "500-5100-0000",
                "hospital-health-care-facility": "800-8500-0000",
                "medical-services": "800-8500-0000",
                "shopping": "600-6600-0000",
                "shopping-mall": "600-6600-0068",
                "atm-bank-exchange": "700-7600-0000",
                "bank": "700-7600-0000",
                "petrol-station": "700-7600-0116",
                "sports-recreation": "800-8400-0000",
                "sights-museums": "300-3000-0000",
                "tourist-attraction": "300-3100-0000"
            }
            
            category_ids = []
            for cat in here_categories:
                if cat in here_cat_mapping:
                    category_ids.append(here_cat_mapping[cat])
            
            if category_ids:
                params["categories"] = ",".join(category_ids)
        
        logger.info(f"HERE Places API request: {places_url}")
        logger.info(f"Params: {params}")
        
        response = requests.get(places_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            
            results = []
            for item in items:
                position = item.get('position', {})
                address = item.get('address', {})
                contacts = item.get('contacts', [])
                
                if position.get('lat') and position.get('lng'):
                    result_lat, result_lng = position['lat'], position['lng']
                    
                    # Check distance
                    distance = geodesic((lat, lon), (result_lat, result_lng)).kilometers
                    
                    # Extract contact info
                    phone = None
                    website = None
                    
                    for contact in contacts:
                        if contact.get('phone'):
                            phone = contact['phone'][0].get('value') if contact['phone'] else None
                        if contact.get('www'):
                            website = contact['www'][0].get('value') if contact['www'] else None
                    
                    # Extract category
                    categories = item.get('categories', [])
                    category = categories[0].get('name', 'place') if categories else 'place'
                    
                    results.append({
                        'lat': result_lat,
                        'lon': result_lng,
                        'name': item.get('title', 'Unknown Place'),
                        'address': address.get('label', 'Address not available'),
                        'category': category,
                        'here_id': item.get('id', ''),
                        'distance_km': round(distance, 2),
                        'phone': phone,
                        'website': website,
                        'score': item.get('scoring', {}).get('queryScore', 0)
                    })
            
            logger.info(f"HERE Places found {len(results)} results")
            return results
        else:
            logger.error(f"HERE Places API error: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        logger.error(f"Error in search_with_here_places: {str(e)}")
        return []

def search_places(search_intent: Dict[str, Any], location: str, radius: int, user_lat: float = None, user_lng: float = None) -> List[LocationResult]:
    """Main search function using HERE Maps APIs"""
    
    try:
        # Determine center coordinates
        if user_lat and user_lng:
            center_lat, center_lon = user_lat, user_lng
            logger.info(f"Using GPS coordinates: {center_lat:.4f}, {center_lon:.4f}")
        else:
            # Use HERE Geocoding to get location coordinates
            geocoding_url = f"https://geocode.search.hereapi.com/v1/geocode"
            params = {
                "q": location,
                "limit": 1,
                "apiKey": HERE_API_KEY
            }
            
            response = requests.get(geocoding_url, params=params, timeout=8)
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                if items:
                    position = items[0].get('position', {})
                    if position.get('lat') and position.get('lng'):
                        center_lat, center_lon = position['lat'], position['lng']
                        logger.info(f"Geocoded '{location}' to: {center_lat:.4f}, {center_lon:.4f}")
                    else:
                        # Default to Jakarta
                        center_lat, center_lon = -6.2088, 106.8456
                        logger.warning(f"Could not geocode '{location}', using Jakarta coordinates")
                else:
                    center_lat, center_lon = -6.2088, 106.8456
                    logger.warning(f"No results for '{location}', using Jakarta coordinates")
            else:
                center_lat, center_lon = -6.2088, 106.8456
                logger.error(f"HERE geocoding failed for '{location}', using Jakarta coordinates")
        
        radius_km = radius / 1000
        logger.info(f"Searching within {radius_km}km radius")
        
        # Try multiple search methods
        all_results = []
        
        # Method 1: HERE Places API (primary)
        logger.info("Trying HERE Places API...")
        places_results = search_with_here_places(search_intent, center_lat, center_lon, radius)
        all_results.extend(places_results)
        
        # Method 2: HERE Geocoding API (if first method doesn't return enough results)
        if len(all_results) < 5:
            logger.info("Adding HERE Geocoding results...")
            geocoding_results = search_with_here_geocoding(search_intent, center_lat, center_lon, radius_km)
            
            # Avoid duplicates
            existing_coords = {(round(r['lat'], 4), round(r['lon'], 4)) for r in all_results}
            for result in geocoding_results:
                result_coord = (round(result['lat'], 4), round(result['lon'], 4))
                if result_coord not in existing_coords:
                    all_results.append(result)
        
        # Process results into LocationResult objects
        processed_results = []
        for result in all_results:
            try:
                # Extract data
                name = result.get('name', 'Unknown Place')
                address = result.get('address', 'Address not available')
                lat = result.get('lat')
                lon = result.get('lon')
                category = result.get('category', 'place')
                here_id = result.get('here_id', '')
                
                # Calculate distance
                distance = geodesic((center_lat, center_lon), (lat, lon)).kilometers
                
                # Extract additional info
                phone = result.get('phone')
                website = result.get('website')
                
                # Create HERE Maps URL
                here_url = f"https://wego.here.com/directions/drive/{center_lat},{center_lon}/{lat},{lon}"
                
                location_result = LocationResult(
                    name=name,
                    address=address,
                    rating=None,  # HERE doesn't provide ratings in basic plan
                    lat=lat,
                    lng=lon,
                    distance=round(distance, 2),
                    category=category,
                    place_id=here_id,
                    phone=phone,
                    website=website,
                    here_url=here_url
                )
                
                processed_results.append(location_result)
                
            except Exception as e:
                logger.warning(f"Error processing result: {str(e)}")
                continue
        
        # Sort by distance and score
        processed_results.sort(key=lambda x: (x.distance or 999, -result.get('score', 0)))
        
        logger.info(f"Final results: {len(processed_results)} locations found")
        return processed_results[:15]  # Limit to 15 results
        
    except Exception as e:
        logger.error(f"Error in search_places: {str(e)}")
        return []

def generate_llm_response(query: str, locations: List[LocationResult], llm: ChatOpenAI) -> str:
    """Generate conversational response about found locations"""
    
    if not locations:
        return "Maaf, saya tidak menemukan lokasi yang sesuai dengan permintaan Anda. Mungkin coba dengan kata kunci yang berbeda atau perluas radius pencarian."
    
    location_summary = []
    for i, loc in enumerate(locations[:5], 1):
        summary = f"{i}. {loc.name}"
        if loc.address and loc.address != "Address not available":
            summary += f" di {loc.address}"
        if loc.distance:
            summary += f" (jarak: {loc.distance} km)"
        summary += f" - kategori: {loc.category}"
        if loc.phone:
            summary += f" | Tel: {loc.phone}"
        location_summary.append(summary)
    
    locations_text = "\n".join(location_summary)
    
    system_prompt = f"""Kamu adalah asisten pencarian lokasi yang ramah dan helpful menggunakan data HERE Maps. 
User menanyakan: "{query}"

Berikut lokasi yang ditemukan:
{locations_text}

Berikan respons dalam Bahasa Indonesia yang:
1. Menyambut pertanyaan dengan ramah
2. Rekomendasikan 2-3 tempat terbaik (terdekat/paling relevan)
3. Sebutkan detail menarik seperti jarak, telepon jika ada
4. Ajak mereka lihat peta untuk detail lebih lanjut
5. Gunakan tone conversational dan helpful
6. Sebutkan bahwa data dari HERE Maps untuk akurasi yang lebih baik

Jangan listing semua - fokus pada rekomendasi terbaik."""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Buatkan respons untuk query: {query}")
        ]
        
        response = llm.invoke(messages)
        return response.content if hasattr(response, 'content') else str(response)
    
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        return f"Saya menemukan {len(locations)} lokasi yang sesuai dengan pencarian '{query}'. Data dari HERE Maps memberikan informasi yang akurat dan terkini. Silakan lihat daftar dan peta di bawah!"

# Routes
@app.get("/")
async def root():
    """Serve the main HTML file"""
    return FileResponse('index.html')

@app.post("/search", response_model=ChatResponse)
async def search_locations(
    query: LocationQuery,
    llm: ChatOpenAI = Depends(get_llm_client)
):
    """Main search endpoint using HERE Maps APIs"""
    
    try:
        logger.info(f"Processing query: {query.query}")
        start_time = time.time()
        
        # Step 1: Extract search intent
        search_intent = extract_search_intent(query.query, llm)
        logger.info(f"Search intent: {search_intent}")
        
        # Step 2: Search with HERE Maps
        locations = search_places(
            search_intent, 
            query.location, 
            query.radius,
            query.lat,
            query.lng
        )
        
        # Step 3: Generate AI response
        llm_response = generate_llm_response(query.query, locations, llm)
        
        # Step 4: Determine map center
        if query.lat and query.lng:
            center_lat, center_lng = query.lat, query.lng
        elif locations:
            center_lat = sum([loc.lat for loc in locations]) / len(locations)
            center_lng = sum([loc.lng for loc in locations]) / len(locations)
        else:
            center_lat, center_lng = -6.2088, 106.8456
        
        processing_time = round(time.time() - start_time, 2)
        logger.info(f"Request processed in {processing_time}s")
        
        return ChatResponse(
            response=llm_response,
            locations=locations,
            map_center={"lat": center_lat, "lng": center_lng},
            search_query=search_intent["keywords"],
            total_found=len(locations)
        )
    
    except Exception as e:
        logger.error(f"Error in search_locations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check with HERE Maps integration status"""
    try:
        if llm_client is None:
            initialize_clients()
        
        # Test OpenRouter
        test_start = time.time()
        test_response = llm_client.invoke("Hello")
        openrouter_latency = round((time.time() - test_start) * 1000, 2)
        
        # Test HERE Maps API
        here_status = "disconnected"
        here_latency = 0
        try:
            test_here_start = time.time()
            test_url = f"https://geocode.search.hereapi.com/v1/geocode?q=jakarta&limit=1&apiKey={HERE_API_KEY}"
            here_response = requests.get(test_url, timeout=5)
            here_latency = round((time.time() - test_here_start) * 1000, 2)
            if here_response.status_code == 200:
                here_status = "connected"
        except Exception as here_error:
            logger.warning(f"HERE Maps test failed: {here_error}")
            here_status = "error"
        
        return {
            "status": "healthy",
            "services": {
                "openrouter": {
                    "status": "connected",
                    "latency_ms": openrouter_latency,
                    "model": os.getenv("MODEL", "deepseek/deepseek-chat-v3")
                },
                "here_maps": {
                    "status": here_status,
                    "latency_ms": here_latency,
                    "apis": [
                        "Geocoding API",
                        "Places API"
                    ]
                }
            },
            "features": [
                "AI-powered query understanding with HERE Maps categories",
                "Enhanced location coverage with HERE Maps data",
                "Brand-specific search with better accuracy", 
                "GPS auto-location detection",
                "Multiple HERE Maps API endpoints for reliability",
                "Phone numbers and website info when available",
                "Indonesian language support",
                "Country-specific search (Indonesia)",
                "Comprehensive POI categories"
            ],
            "improvements": [
                "Replaced Mapbox with HERE Maps for better coverage",
                "Added HERE Geocoding and Places APIs",
                "Enhanced category mapping for Indonesian POIs",
                "Better address formatting and place information",
                "Added phone and website data extraction",
                "Improved search accuracy for local businesses"
            ],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": "3.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/test-here")
async def test_here_apis():
    """Test HERE Maps APIs with sample queries"""
    try:
        test_results = {}
        
        # Test Geocoding API
        logger.info("Testing HERE Geocoding API...")
        geocoding_url = f"https://geocode.search.hereapi.com/v1/geocode"
        params = {
            "q": "restaurant jakarta",
            "at": "-6.2088,106.8456",
            "limit": 5,
            "apiKey": HERE_API_KEY
        }
        
        start_time = time.time()
        response = requests.get(geocoding_url, params=params, timeout=10)
        geocoding_latency = round((time.time() - start_time) * 1000, 2)
        
        if response.status_code == 200:
            data = response.json()
            test_results["geocoding_api"] = {
                "status": "success",
                "latency_ms": geocoding_latency,
                "results_count": len(data.get('items', [])),
                "sample_result": data.get('items', [{}])[0].get('title', 'N/A') if data.get('items') else None
            }
        else:
            test_results["geocoding_api"] = {
                "status": "error",
                "error": f"HTTP {response.status_code}",
                "latency_ms": geocoding_latency
            }
        
        # Test Places API
        logger.info("Testing HERE Places API...")
        places_url = "https://discover.search.hereapi.com/v1/discover"
        places_params = {
            "at": "-6.2088,106.8456",
            "q": "mcdonald",
            "limit": 5,
            "apiKey": HERE_API_KEY
        }
        
        start_time = time.time()
        places_response = requests.get(places_url, params=places_params, timeout=10)
        places_latency = round((time.time() - start_time) * 1000, 2)
        
        if places_response.status_code == 200:
            places_data = places_response.json()
            test_results["places_api"] = {
                "status": "success",
                "latency_ms": places_latency,
                "results_count": len(places_data.get('items', [])),
                "sample_result": places_data.get('items', [{}])[0].get('title', 'N/A') if places_data.get('items') else None
            }
        else:
            test_results["places_api"] = {
                "status": "error",
                "error": f"HTTP {places_response.status_code}",
                "latency_ms": places_latency
            }
        
        return {
            "here_api_key_status": "configured" if HERE_API_KEY else "missing",
            "test_results": test_results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
    except Exception as e:
        logger.error(f"HERE Maps test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"HERE Maps test failed: {str(e)}")

@app.get("/test-search-integration")
async def test_complete_search():
    """Test complete search flow with HERE Maps integration"""
    try:
        if llm_client is None:
            initialize_clients()
        
        test_queries = [
            {"query": "McDonald's terdekat", "location": "Jakarta"},
            {"query": "tempat makan", "location": "Jakarta"},
            {"query": "kolam renang", "location": "Jakarta"},
            {"query": "rumah sakit", "location": "Jakarta"}
        ]
        
        results = {}
        
        for test_case in test_queries:
            query_text = test_case["query"]
            try:
                start_time = time.time()
                
                # Test intent extraction
                search_intent = extract_search_intent(query_text, llm_client)
                
                # Test search
                locations = search_places(
                    search_intent, 
                    test_case["location"], 
                    5000,  # 5km radius
                    -6.2088,  # Jakarta lat
                    106.8456  # Jakarta lng
                )
                
                total_time = round((time.time() - start_time) * 1000, 2)
                
                results[query_text] = {
                    "status": "success",
                    "intent": search_intent,
                    "locations_found": len(locations),
                    "processing_time_ms": total_time,
                    "sample_locations": [
                        {
                            "name": loc.name,
                            "address": loc.address,
                            "distance_km": loc.distance,
                            "category": loc.category
                        } for loc in locations[:3]
                    ]
                }
                
            except Exception as e:
                results[query_text] = {
                    "status": "error",
                    "error": str(e),
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                }
        
        return {
            "test_results": results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
    except Exception as e:
        logger.error(f"Complete search test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search integration test failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Check environment variables
    required_env_vars = ["OPENROUTER_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please set the following environment variables:")
        for var in missing_vars:
            logger.info(f"- {var}")
        sys.exit(1)
    
    # Check HERE API key
    if not HERE_API_KEY:
        logger.warning("HERE_API_KEY not configured - some features may be limited")
    else:
        logger.info("HERE Maps integration enabled")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("Application started successfully with HERE Maps integration")