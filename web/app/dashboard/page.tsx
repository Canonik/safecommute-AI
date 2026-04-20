import Link from "next/link";
import { supabaseServer } from "@/lib/supabase/server";
import { PageHeading } from "@/components/dashboard/page-heading";
import { StatBlock } from "@/components/dashboard/stat-block";
import { JobCard, type JobStatus } from "@/components/dashboard/job-card";
import { PRICE_PER_RUN_EUR, PRICE_SUBSCRIPTION_EUR } from "@/lib/utils";

export const dynamic = "force-dynamic";

export default async function DashboardHome() {
  const supabase = supabaseServer();
  const { data: auth } = await supabase.auth.getUser();
  const user = auth.user!;

  const [{ data: ent }, { count: siteCount }, { count: clipCount }, { data: jobs }] = await Promise.all([
    supabase
      .from("entitlements")
      .select("subscription_active, per_run_credits")
      .eq("owner", user.id)
      .maybeSingle(),
    supabase.from("sites").select("id", { count: "exact", head: true }).eq("owner", user.id),
    supabase.from("audio_clips").select("id", { count: "exact", head: true }).eq("owner", user.id),
    supabase
      .from("finetune_jobs")
      .select("id, site_id, status, model_path, error, created_at, sites(name)")
      .eq("owner", user.id)
      .order("created_at", { ascending: false })
      .limit(5),
  ]);

  const hasSub = ent?.subscription_active ?? false;
  const credits = ent?.per_run_credits ?? 0;

  return (
    <div>
      <PageHeading eyebrow="Overview" title="Your calibration" accent="red">
        <Link
          href="/dashboard/sites/new"
          className="border-3 border-ink bg-ink text-paper px-5 py-3 font-display uppercase text-sm tracking-tight hover:bg-bauhaus-red transition-colors"
        >
          + New site
        </Link>
      </PageHeading>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
        <StatBlock value={hasSub ? "ON" : "OFF"} label="Subscription" color={hasSub ? "blue" : "paper"} />
        <StatBlock value={String(credits)} label="Per-run credits" color={credits > 0 ? "yellow" : "paper"} delay={0.05} />
        <StatBlock value={String(siteCount ?? 0)} label="Sites" color="paper" delay={0.1} />
        <StatBlock value={String(clipCount ?? 0)} label="Audio clips" color="ink" delay={0.15} />
      </div>

      {!hasSub && credits === 0 && (
        <div className="border-3 border-ink bg-bauhaus-yellow p-6 md:p-8 mb-10 shadow-[6px_6px_0_0_#0a0a0a]">
          <div className="grid md:grid-cols-[1fr_auto] gap-6 items-center">
            <div>
              <div className="font-mono text-[11px] uppercase tracking-widest mb-2">No credits</div>
              <h2 className="font-display uppercase text-3xl md:text-4xl leading-[0.95]">
                Unlock fine-tuning
              </h2>
              <p className="font-grotesk text-sm mt-3 max-w-md">
                Start a €{PRICE_SUBSCRIPTION_EUR} subscription for unlimited tunes, or buy one run
                for €{PRICE_PER_RUN_EUR}. Either option unlocks uploads and the run button.
              </p>
            </div>
            <Link
              href="/dashboard/billing"
              className="justify-self-start md:justify-self-end border-3 border-ink bg-ink text-paper px-5 py-3 font-display uppercase text-sm tracking-tight hover:bg-bauhaus-red transition-colors"
            >
              Go to billing →
            </Link>
          </div>
        </div>
      )}

      <div className="mb-3 flex items-center gap-3 font-mono text-[11px] uppercase tracking-widest">
        <span className="inline-block h-3 w-3 bg-bauhaus-blue" />
        <span>Recent jobs</span>
      </div>
      {jobs && jobs.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {jobs.map((j) => {
            type JobRow = {
              id: string;
              site_id: string;
              status: JobStatus;
              model_path: string | null;
              error: string | null;
              created_at: string;
              sites: { name: string } | { name: string }[] | null;
            };
            const row = j as unknown as JobRow;
            const siteName = Array.isArray(row.sites)
              ? row.sites[0]?.name ?? "—"
              : row.sites?.name ?? "—";
            return (
              <JobCard
                key={row.id}
                id={row.id}
                status={row.status}
                siteName={siteName}
                createdAt={row.created_at}
                modelPath={row.model_path}
                error={row.error}
              />
            );
          })}
        </div>
      ) : (
        <div className="border-3 border-ink border-dashed p-10 text-center">
          <div className="font-grotesk">
            No jobs yet. Create a site, upload audio, and press{" "}
            <span className="font-display">Run fine-tune</span>.
          </div>
        </div>
      )}
    </div>
  );
}
