import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const MAILTO_PILOT =
  "mailto:alessandro.canonico2@studbocconi.it?subject=SafeCommute%20AI%20%E2%80%94%20calibration%20pilot%20inquiry&body=Organization%3A%20%0ASite%20count%3A%20%0AVertical%20(transit%2Fretail%2Fschool%2Feldercare%2Findustrial)%3A%20%0ANotes%3A%20";

export const MAILTO_CONTACT = "mailto:alessandro.canonico2@studbocconi.it";

export const GITHUB_URL = "https://github.com/Canonik/safecommute-AI";

export const DEMO_DOWNLOAD_URL = "/demo/safecommute-v2-demo.zip";

export const DASHBOARD_URL = "/dashboard";
export const SIGN_IN_URL = "/sign-in";

export const PRICE_SUBSCRIPTION_EUR = 10;
export const PRICE_PER_RUN_EUR = 3;
