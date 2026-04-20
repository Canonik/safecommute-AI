import { redirect } from "next/navigation";
import type { ReactNode } from "react";
import { isSupabaseConfigured, supabaseServer } from "@/lib/supabase/server";
import { DashboardShell } from "@/components/dashboard/dashboard-shell";
import { DashboardUnavailable } from "@/components/dashboard/dashboard-unavailable";

export const dynamic = "force-dynamic";

export default async function DashboardLayout({ children }: { children: ReactNode }) {
  if (!isSupabaseConfigured()) {
    return <DashboardUnavailable />;
  }
  const supabase = supabaseServer();
  const { data } = await supabase.auth.getUser();
  if (!data.user) {
    redirect("/sign-in?next=/dashboard");
  }
  return <DashboardShell userEmail={data.user.email ?? null}>{children}</DashboardShell>;
}
