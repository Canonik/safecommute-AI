import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const MAILTO_PILOT =
  "mailto:alessandro.canonico2@studbocconi.it?subject=SafeCommute%20AI%20%E2%80%94%20calibration%20pilot%20inquiry&body=Organization%3A%20%0ASite%20count%3A%20%0AVertical%20(transit%2Fretail%2Fschool%2Feldercare%2Findustrial)%3A%20%0ANotes%3A%20";

export const MAILTO_CONTACT = "mailto:alessandro.canonico2@studbocconi.it";

export const GITHUB_URL = "https://github.com/Canonik/safecommute-AI";
