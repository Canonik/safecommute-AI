import Link from "next/link";
import { supabaseServer } from "@/lib/supabase/server";
import { PageHeading } from "@/components/dashboard/page-heading";
import { SiteGrid } from "@/components/dashboard/site-grid";

export const dynamic = "force-dynamic";

export default async function SitesPage() {
  const supabase = supabaseServer();
  const { data: auth } = await supabase.auth.getUser();
  const user = auth.user!;

  const { data: sites } = await supabase
    .from("sites")
    .select("id, name, environment, created_at")
    .eq("owner", user.id)
    .order("created_at", { ascending: false });

  return (
    <div>
      <PageHeading eyebrow="Sites" title="Your sites" accent="yellow">
        <Link
          href="/dashboard/sites/new"
          className="border-3 border-ink bg-ink text-paper px-5 py-3 font-display uppercase text-sm tracking-tight hover:bg-bauhaus-red transition-colors"
        >
          + New site
        </Link>
      </PageHeading>
      <SiteGrid sites={sites ?? []} />
    </div>
  );
}
