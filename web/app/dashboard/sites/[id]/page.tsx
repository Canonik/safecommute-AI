import { notFound } from "next/navigation";
import { supabaseServer } from "@/lib/supabase/server";
import { PageHeading } from "@/components/dashboard/page-heading";
import { SiteWorkspace } from "@/components/dashboard/site-workspace";
import type { JobStatus } from "@/components/dashboard/job-card";

export const dynamic = "force-dynamic";

export default async function SiteDetailPage({ params }: { params: { id: string } }) {
  const supabase = supabaseServer();
  const { data: auth } = await supabase.auth.getUser();
  const user = auth.user!;

  const { data: site } = await supabase
    .from("sites")
    .select("id, name, environment, created_at")
    .eq("id", params.id)
    .eq("owner", user.id)
    .maybeSingle();
  if (!site) notFound();

  const [{ data: clips }, { data: jobs }, { data: ent }] = await Promise.all([
    supabase
      .from("audio_clips")
      .select("id, filename, size_bytes, duration_s, uploaded_at")
      .eq("site_id", params.id)
      .order("uploaded_at", { ascending: false }),
    supabase
      .from("finetune_jobs")
      .select("id, status, model_path, error, created_at")
      .eq("site_id", params.id)
      .order("created_at", { ascending: false })
      .limit(5),
    supabase
      .from("entitlements")
      .select("subscription_active, per_run_credits")
      .eq("owner", user.id)
      .maybeSingle(),
  ]);

  const hasCredit = Boolean(ent?.subscription_active) || (ent?.per_run_credits ?? 0) > 0;

  return (
    <div>
      <PageHeading eyebrow={site.environment} title={site.name} accent="red" />
      <SiteWorkspace
        site={site}
        clips={clips ?? []}
        jobs={(jobs ?? []) as Array<{
          id: string;
          status: JobStatus;
          model_path: string | null;
          error: string | null;
          created_at: string;
        }>}
        hasCredit={hasCredit}
        subscriptionActive={Boolean(ent?.subscription_active)}
        perRunCredits={ent?.per_run_credits ?? 0}
      />
    </div>
  );
}
