import type { ReactNode } from "react";
import { COLORS } from "./console-data";

/** A pane with a rounded border and its title on the top edge. */
export const Pane = ({ title, focused, footer, children }: { title: string; focused: boolean; footer?: string; children: ReactNode }) => (
  <div className="relative h-full rounded-md border px-2 pt-2.5 pb-1" style={{ borderColor: focused ? COLORS.accent : COLORS.dim }}>
    <span className="absolute -top-[12px] left-2 bg-[#16161a] font-bold text-white">{title}</span>
    {footer && (
      <span className="absolute -bottom-[12px] right-2 bg-[#16161a]" style={{ color: COLORS.dim }}>
        {` ${footer} `}
      </span>
    )}
    <div className="h-full overflow-hidden">{children}</div>
  </div>
);
