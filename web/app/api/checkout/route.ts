import { NextResponse, type NextRequest } from "next/server";
import { z } from "zod";
import { supabaseServer } from "@/lib/supabase/server";
import {
  isStripeConfigured,
  publicSiteUrl,
  STRIPE_PRICE_PER_RUN,
  STRIPE_PRICE_SUBSCRIPTION,
  stripe,
} from "@/lib/stripe";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const schema = z.object({
  kind: z.enum(["subscription", "per_run"]),
});

export async function POST(request: NextRequest) {
  if (!isStripeConfigured()) {
    return NextResponse.json(
      { error: "Stripe is not configured on this deployment." },
      { status: 501 }
    );
  }

  const supabase = supabaseServer();
  const { data: auth } = await supabase.auth.getUser();
  const user = auth.user;
  if (!user) {
    return NextResponse.json({ error: "Not authenticated." }, { status: 401 });
  }

  const parsed = schema.safeParse(await request.json().catch(() => ({})));
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid body." }, { status: 400 });
  }
  const { kind } = parsed.data;

  const priceId = kind === "subscription" ? STRIPE_PRICE_SUBSCRIPTION : STRIPE_PRICE_PER_RUN;
  if (!priceId) {
    return NextResponse.json(
      { error: `Missing Stripe price id for ${kind}. Set STRIPE_PRICE_${kind.toUpperCase()}.` },
      { status: 501 }
    );
  }

  const site = publicSiteUrl();
  const session = await stripe().checkout.sessions.create({
    mode: "payment",
    customer_email: user.email ?? undefined,
    line_items: [{ price: priceId, quantity: 1 }],
    success_url: `${site}/dashboard/billing?status=success&session_id={CHECKOUT_SESSION_ID}`,
    cancel_url: `${site}/dashboard/billing?status=cancel`,
    metadata: { owner: user.id, kind },
    payment_intent_data: { metadata: { owner: user.id, kind } },
  });

  return NextResponse.json({ url: session.url });
}
