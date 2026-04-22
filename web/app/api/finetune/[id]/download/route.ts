import { NextResponse, type NextRequest } from "next/server";
import { supabaseServer, supabaseService } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

// /api/finetune/[id]/download?file=model|thresholds|report
//
// Returns a short-lived (60s) signed URL for one of the three artefacts the
// worker uploads to the `models-deliverable` bucket on a successful job:
//
//   {owner}/{job_id}/model.onnx               — the INT8 ONNX deployable
//   {owner}/{job_id}/thresholds.json          — Youden / F1 / low_fpr / low_fpr_site
//   {owner}/{job_id}/deployment_report.json   — test_deployment.py output
//
// Authentication: the caller must own the job (`finetune_jobs.owner = auth.uid()`)
// AND the job must be in `status='succeeded'` with a non-null `model_path`.
// Anything else returns 404 so we don't leak existence of jobs across accounts.

const FILES: Record<string, { name: string; contentType: string }> = {
  model: { name: "model.onnx", contentType: "application/octet-stream" },
  thresholds: { name: "thresholds.json", contentType: "application/json" },
  report: { name: "deployment_report.json", contentType: "application/json" },
};

const DELIVERABLE_BUCKET = "models-deliverable";
const SIGNED_URL_TTL_S = 60;

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const supabase = supabaseServer();
  const { data: auth } = await supabase.auth.getUser();
  const user = auth.user;
  if (!user) {
    return NextResponse.json({ error: "unauthenticated" }, { status: 401 });
  }

  const fileKey = request.nextUrl.searchParams.get("file") ?? "model";
  const file = FILES[fileKey];
  if (!file) {
    return NextResponse.json(
      { error: `invalid file kind — expected one of: ${Object.keys(FILES).join(", ")}` },
      { status: 400 }
    );
  }

  const { data: job, error: jobErr } = await supabase
    .from("finetune_jobs")
    .select("id, owner, status, model_path")
    .eq("id", params.id)
    .eq("owner", user.id)
    .maybeSingle();
  if (jobErr || !job) {
    // Indistinguishable from "not yours" so callers can't enumerate jobs.
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }
  if (job.status !== "succeeded" || !job.model_path) {
    return NextResponse.json(
      { error: `job status is ${job.status}, artefacts not available` },
      { status: 409 }
    );
  }

  // model_path is the directory prefix "{owner}/{job_id}" the worker wrote.
  // Concatenating the filename gives the exact object path.
  const objectPath = `${job.model_path}/${file.name}`;

  const db = supabaseService();
  const { data: signed, error: signErr } = await db.storage
    .from(DELIVERABLE_BUCKET)
    .createSignedUrl(objectPath, SIGNED_URL_TTL_S, {
      download: file.name,
    });
  if (signErr || !signed) {
    return NextResponse.json(
      { error: signErr?.message ?? "could not sign download" },
      { status: 500 }
    );
  }

  // 302 redirect to the signed URL so a plain <a href> click works. Clients
  // that want JSON can hit `?file=...&format=json` for the raw signed URL.
  if (request.nextUrl.searchParams.get("format") === "json") {
    return NextResponse.json({
      signed_url: signed.signedUrl,
      expires_in_s: SIGNED_URL_TTL_S,
      content_type: file.contentType,
      filename: file.name,
    });
  }
  return NextResponse.redirect(signed.signedUrl, { status: 302 });
}
