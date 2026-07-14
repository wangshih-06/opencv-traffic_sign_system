import type { HTMLAttributes, ReactNode } from "react";
import clsx from "clsx";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  title?: string;
  eyebrow?: string;
  action?: ReactNode;
  padded?: boolean;
}

export function Card({
  title,
  eyebrow,
  action,
  padded = true,
  className,
  children,
  ...props
}: CardProps) {
  return (
    <section className={clsx("card", !padded && "card--flush", className)} {...props}>
      {(title || eyebrow || action) && (
        <header className="card__header">
          <div>
            {eyebrow && <span className="eyebrow">{eyebrow}</span>}
            {title && <h2>{title}</h2>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}
