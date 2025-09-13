# test_debug.py
import requests
import json
import sys

def test_endpoints():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing LLM Location Assistant...")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health Check Passed")
            print(f"   OpenRouter: {health_data['services']['openrouter']['status']}")
            print(f"   OpenRouter Latency: {health_data['services']['openrouter']['latency_ms']}ms")
            print(f"   Nominatim: {health_data['services']['nominatim']['status']}")
        else:
            print(f"❌ Health Check Failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Health Check Error: {str(e)}")
        return False
    
    # Test 2: AI Connection
    print("\n2️⃣ Testing AI Connection...")
    try:
        response = requests.get(f"{base_url}/test-ai", timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            ai_data = response.json()
            print(f"✅ AI Connection Passed")
            print(f"   Model: {ai_data['model']}")
            print(f"   Latency: {ai_data['latency_ms']}ms")
            print(f"   Response: {ai_data['response'][:100]}...")
        else:
            print(f"❌ AI Connection Failed: {response.text}")
    except Exception as e:
        print(f"❌ AI Connection Error: {str(e)}")
    
    # Test 3: Simple Search
    print("\n3️⃣ Testing Simple Search...")
    try:
        search_data = {
            "query": "cafe jakarta",
            "location": "Jakarta, Indonesia",
            "radius": 2000
        }
        
        response = requests.post(
            f"{base_url}/search",
            headers={"Content-Type": "application/json"},
            json=search_data,
            timeout=60
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Search Passed")
            print(f"   Total Found: {result['total_found']}")
            print(f"   Search Query: {result['search_query']}")
            print(f"   AI Response: {result['response'][:100]}...")
            
            if result['locations']:
                print(f"   First Location: {result['locations'][0]['name']}")
                print(f"   Address: {result['locations'][0]['address']}")
                print(f"   Distance: {result['locations'][0].get('distance', 'N/A')} km")
        else:
            print(f"❌ Search Failed: {response.text}")
    except Exception as e:
        print(f"❌ Search Error: {str(e)}")
    
    # Test 4: Complex Search
    print("\n4️⃣ Testing Complex Search...")
    try:
        search_data = {
            "query": "restoran padang enak murah",
            "location": "Jakarta, Indonesia", 
            "radius": 5000
        }
        
        response = requests.post(
            f"{base_url}/search",
            headers={"Content-Type": "application/json"},
            json=search_data,
            timeout=60
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Complex Search Passed")
            print(f"   Total Found: {result['total_found']}")
            print(f"   AI interpreted as: {result['search_query']}")
        else:
            print(f"❌ Complex Search Failed: {response.text}")
    except Exception as e:
        print(f"❌ Complex Search Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🏁 Testing Complete!")
    print("\n💡 Tips:")
    print("   - If OSM ID errors persist, they're just warnings")
    print("   - The app should still work and return results")
    print("   - Try the web interface at http://localhost:8000")

def test_overpass_api():
    """Test Overpass API directly"""
    print("\n🔧 Testing Overpass API directly...")
    
    query = """
    [out:json][timeout:25];
    (
      node["amenity"="restaurant"](around:2000,-6.2088,106.8456);
    );
    out body;
    """
    
    try:
        response = requests.post(
            "http://overpass-api.de/api/interpreter",
            data=query,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            elements = data.get('elements', [])
            print(f"✅ Overpass API works - Found {len(elements)} restaurants")
            if elements:
                first = elements[0]
                print(f"   Sample: {first.get('tags', {}).get('name', 'Unnamed')}")
                print(f"   OSM ID type: {type(first.get('id'))}")
        else:
            print(f"❌ Overpass API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Overpass API error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "overpass":
        test_overpass_api()
    else:
        test_endpoints()