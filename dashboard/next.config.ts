import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // the dashboard is a pure client of the engine's JSON/SSE API, nothing fancy
  reactStrictMode: true,
};

export default nextConfig;
