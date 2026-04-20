import { NextResponse, type NextRequest } from "next/server";
import { z } from "zod";
import { supabaseServer, supabaseService } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const schema = z.object({ site_id: z.string().uuid() });

export async function POST(request: NextRequest) {
  const supabase = supabaseServer();
  const { data: auth } = await supabase.auth.getUser();
  const user = auth.user;
  if (!user) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });

  const parsed = schema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) {
    return NextResponse.json({ error: "invalid body" }, { status: 400 });
  }
  const { site_id } = parsed.data;

  const { data: site } = await supabase
    .from("sites")
    .select("id, owner")
    .eq("id", site_id)
    .maybeSingle();
  if (!site || site.owner !== user.id) {
    return NextResponse.json({ error: "site not found" }, { status: 404 });
  }

  const { count: clipCount } = await supabase
    .from("audio_clips")
    .select("id", { count: "exact", head: true })
    .eq("site_id", site_id);
  if (!clipCount || clipCount < 3) {
    return NextResponse.json(
      { error: `need at least 3 audio clips — have ${clipCount ?? 0}` },
      { status: 422 }
    );
  }

  const { data: ent } = await supabase
    .from("entitlements")
    .select("subscription_active, per_run_credits")
    .eq("owner", user.id)
    .maybeSingle();
  const hasSub = ent?.subscription_active ?? false;
  const credits = ent?.per_run_credits ?? 0;
  if (!hasSub && credits < 1) {
    return NextResponse.json(
      { error: "no credit — buy a subscription or a per-run credit first" },
      { status: 402 }
    );
  }

  const db = supabaseService();

  if (!hasSub) {
    await db.from("entitlements").update({ per_run_credits: credits - 1 }).eq("owner", user.id);
  }

  const { data: job, error } = await db
    .from("finetune_jobs")
    .insert({ site_id, owner: user.id, status: "queued" })
    .select("id")
    .single();
  if (error || !job) {
    return NextResponse.json(
      { error: error?.message ?? "could not queue job" },
      { status: 500 }
    );
  }
  return NextResponse.json({ job_id: job.id, status: "queued" });
}
