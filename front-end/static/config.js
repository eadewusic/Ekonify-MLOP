// config.js - Configuration for Ekonify frontend
const config = {
  // Use the hostname to determine the correct API URL
  apiBaseUrl:
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
      ? "http://localhost:8000"
      : "https://your-railway-app-name.railway.app", // Replace with your actual Railway API URL once deployed
};
