import Link from "next/link";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";

export type Crumb = { label: string; href?: string };

export default function PageHeader({ crumbs, title, children }: { crumbs: Crumb[]; title: string; children?: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2">
      <Breadcrumb>
        <BreadcrumbList>
          {crumbs.map((crumb, i) => (
            <span key={crumb.label} className="contents">
              {i > 0 ? <BreadcrumbSeparator /> : null}
              <BreadcrumbItem>
                {crumb.href === undefined ? (
                  <BreadcrumbPage>{crumb.label}</BreadcrumbPage>
                ) : (
                  <BreadcrumbLink asChild>
                    <Link href={crumb.href}>{crumb.label}</Link>
                  </BreadcrumbLink>
                )}
              </BreadcrumbItem>
            </span>
          ))}
        </BreadcrumbList>
      </Breadcrumb>
      <h1 className="font-mono text-xl font-semibold break-all">{title}</h1>
      {children === undefined ? null : <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">{children}</div>}
    </div>
  );
}
