/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  // Remove static export for Cloud Run deployment
  // output: 'export',
  // trailingSlash: true,
  // distDir: 'dist',
  images: {
    unoptimized: true
  }
}

module.exports = nextConfig