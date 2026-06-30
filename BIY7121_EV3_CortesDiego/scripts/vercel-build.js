const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const vizDir = path.join(root, "visualizacion");
const apiUrl =
  process.env.PREDICT_API_URL || "http://127.0.0.1:5050/predict";

const configJs = `window.PREDICT_API_URL = ${JSON.stringify(apiUrl)};\n`;
fs.writeFileSync(path.join(vizDir, "config.js"), configJs, "utf8");

console.log("Vercel build OK — PREDICT_API_URL =", apiUrl);
