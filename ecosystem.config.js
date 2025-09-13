module.exports = {
  apps: [
    {
      name: "ai-location-assistant", // bebas, nama aplikasinya
      script: "uvicorn",
      args: "main:app --host 0.0.0.0 --port 8110",
      interpreter: "python3",
      cwd: "/root/test", // path project kamu
      env: {
        ENVIRONMENT: "production",
      },
    },
  ],
};
