import { redirect } from "next/navigation";
import Link from "next/link";
import { isSupabaseConfigured, supabaseServer } from "@/lib/supabase/server";
import { SignInForm } from "@/components/dashboard/sign-in-form";
import { DashboardUnavailable } from "@/components/dashboard/dashboard-unavailable";

export const dynamic = "force-dynamic";

export default async function SignInPage({
  searchParams,
}: {
  searchParams?: { next?: string };
}) {
  if (!isSupabaseConfigured()) {
    return <DashboardUnavailable />;
  }
  const supabase = supabaseServer();
  const { data } = await supabase.auth.getUser();
  if (data.user) {
    redirect(searchParams?.next ?? "/dashboard");
  }
  return (
    <main className="min-h-screen bg-paper text-ink flex items-center justify-center px-4 py-16">
      <div className="w-full max-w-xl">
        <Link
          href="/"
          className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest mb-6 hover:text-bauhaus-red"
        >
          ← Back to SafeCommute
        </Link>
        <SignInForm nextPath={searchParams?.next ?? "/dashboard"} />
      </div>
    </main>
  );
}
