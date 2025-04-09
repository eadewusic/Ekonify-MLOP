// config.js - Configuration for Ekonify frontend
const config = {
  // Use the hostname to determine the correct API URL
  apiBaseUrl:
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
      ? "http://localhost:8000"
      : "https://ekonify-api.onrender.com", // Replace with your actual Render API URL once deployed
};
