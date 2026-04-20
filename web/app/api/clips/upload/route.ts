import { NextResponse, type NextRequest } from "next/server";
import { z } from "zod";
import { supabaseServer, supabaseService } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const schema = z.object({
  site_id: z.string().uuid(),
  filename: z.string().min(1).max(200),
  content_type: z.string().min(1).max(100),
  size_bytes: z.number().int().positive().max(26214400),
});

// Returns a short-lived signed upload URL scoped to `{owner}/{site_id}/{uuid}-{filename}`.
// Client PUTs directly to Supabase Storage — server never handles the bytes.
export async function POST(request: NextRequest) {
  const supabase = supabaseServer();
  const { data: auth } = await supabase.auth.getUser();
  const user = auth.user;
  if (!user) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });

  const parsed = schema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) {
    return NextResponse.json({ error: "invalid body", issues: parsed.error.issues }, { status: 400 });
  }
  const { site_id, filename, content_type, size_bytes } = parsed.data;

  // Confirm the site belongs to this user.
  const { data: site } = await supabase.from("sites").select("id, owner").eq("id", site_id).maybeSingle();
  if (!site || site.owner !== user.id) {
    return NextResponse.json({ error: "site not found" }, { status: 404 });
  }

  const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, "_").slice(0, 120);
  const objectPath = `${user.id}/${site_id}/${crypto.randomUUID()}-${safeName}`;

  const db = supabaseService();
  const { data, error } = await db.storage.from("audio-uploads").createSignedUploadUrl(objectPath);
  if (error || !data) {
    return NextResponse.json(
      { error: error?.message ?? "could not sign upload" },
      { status: 500 }
    );
  }

  // Pre-record the clip row as `pending`. Client calls /api/clips/confirm after PUT.
  await supabase.from("audio_clips").insert({
    site_id,
    owner: user.id,
    storage_path: objectPath,
    filename: safeName,
    size_bytes,
  });

  return NextResponse.json({
    upload_url: data.signedUrl,
    token: data.token,
    path: objectPath,
    content_type,
  });
}
