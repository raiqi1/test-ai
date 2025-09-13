module.exports = {
  apps: [
    {
      name: "ai-location-assistant",
      script: "uvicorn",
      args: "main:app --host 0.0.0.0 --port 8110",
      interpreter: "python3",
      cwd: "/root/test",
      env: {
        ENVIRONMENT: "production",
      },
    },
  ],
};
