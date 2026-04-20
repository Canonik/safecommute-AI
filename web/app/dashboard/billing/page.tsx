import { supabaseServer } from "@/lib/supabase/server";
import { PageHeading } from "@/components/dashboard/page-heading";
import { BillingPanel } from "@/components/dashboard/billing-panel";

export const dynamic = "force-dynamic";

export default async function BillingPage({
  searchParams,
}: {
  searchParams?: { status?: string };
}) {
  const supabase = supabaseServer();
  const { data: auth } = await supabase.auth.getUser();
  const user = auth.user!;

  const [{ data: ent }, { data: payments }] = await Promise.all([
    supabase
      .from("entitlements")
      .select("subscription_active, per_run_credits, updated_at")
      .eq("owner", user.id)
      .maybeSingle(),
    supabase
      .from("payments")
      .select("kind, amount_cents, currency, status, created_at")
      .eq("owner", user.id)
      .order("created_at", { ascending: false })
      .limit(10),
  ]);

  return (
    <div>
      <PageHeading eyebrow="Billing" title="Pay & unlock" accent="yellow" />
      <BillingPanel
        subscriptionActive={Boolean(ent?.subscription_active)}
        perRunCredits={ent?.per_run_credits ?? 0}
        payments={payments ?? []}
        resultStatus={searchParams?.status ?? null}
      />
    </div>
  );
}
