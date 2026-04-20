import { PageHeading } from "@/components/dashboard/page-heading";
import { NewSiteForm } from "@/components/dashboard/new-site-form";

export const dynamic = "force-dynamic";

export default function NewSitePage() {
  return (
    <div>
      <PageHeading eyebrow="New site" title="Name your site" accent="blue" />
      <NewSiteForm />
    </div>
  );
}
