import { NextResponse, type NextRequest } from "next/server";
import type Stripe from "stripe";
import { isStripeConfigured, stripe } from "@/lib/stripe";
import { supabaseService } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  if (!isStripeConfigured()) {
    return NextResponse.json({ error: "Stripe not configured" }, { status: 501 });
  }
  const secret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!secret) {
    return NextResponse.json({ error: "STRIPE_WEBHOOK_SECRET not set" }, { status: 501 });
  }

  const sig = request.headers.get("stripe-signature");
  if (!sig) return NextResponse.json({ error: "missing signature" }, { status: 400 });

  const raw = await request.text();

  let event: Stripe.Event;
  try {
    event = stripe().webhooks.constructEvent(raw, sig, secret);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: `invalid signature: ${msg}` }, { status: 400 });
  }

  if (event.type !== "checkout.session.completed") {
    return NextResponse.json({ received: true });
  }

  const session = event.data.object as Stripe.Checkout.Session;
  const owner = session.metadata?.owner;
  const kind = session.metadata?.kind as "subscription" | "per_run" | undefined;
  if (!owner || !kind) {
    return NextResponse.json({ error: "missing owner/kind metadata" }, { status: 400 });
  }

  const db = supabaseService();

  await db.from("payments").upsert(
    {
      owner,
      stripe_session_id: session.id,
      kind,
      amount_cents: session.amount_total ?? 0,
      currency: session.currency ?? "eur",
      status: "paid",
    },
    { onConflict: "stripe_session_id" }
  );

  if (kind === "subscription") {
    await db
      .from("entitlements")
      .upsert({ owner, subscription_active: true, updated_at: new Date().toISOString() });
  } else {
    const { data: current } = await db
      .from("entitlements")
      .select("per_run_credits")
      .eq("owner", owner)
      .maybeSingle();
    const next = (current?.per_run_credits ?? 0) + 1;
    await db
      .from("entitlements")
      .upsert({ owner, per_run_credits: next, updated_at: new Date().toISOString() });
  }

  return NextResponse.json({ received: true });
}
