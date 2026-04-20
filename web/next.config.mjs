/** @type {import('next').NextConfig} */
const nextConfig = {
  // Dashboard + API routes require the Node runtime on Vercel.
  // (Previously this was `output: "export"` — static export can't serve /api/*.)
  images: { unoptimized: true },
};

export default nextConfig;
