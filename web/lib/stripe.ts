import Stripe from "stripe";

const STRIPE_SECRET = process.env.STRIPE_SECRET_KEY;

export function isStripeConfigured(): boolean {
  return Boolean(STRIPE_SECRET);
}

let client: Stripe | null = null;
export function stripe(): Stripe {
  if (!STRIPE_SECRET) {
    throw new Error("STRIPE_SECRET_KEY is not set.");
  }
  if (!client) {
    client = new Stripe(STRIPE_SECRET, { apiVersion: "2025-02-24.acacia" });
  }
  return client;
}

export const STRIPE_PRICE_SUBSCRIPTION = process.env.STRIPE_PRICE_SUBSCRIPTION ?? "";
export const STRIPE_PRICE_PER_RUN = process.env.STRIPE_PRICE_PER_RUN ?? "";

export function publicSiteUrl(): string {
  return process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";
}
