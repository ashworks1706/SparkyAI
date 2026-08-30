type BrandMarkProps = {
  className?: string;
  decorative?: boolean;
};

type BrandLogoProps = {
  className?: string;
  markClassName?: string;
  wordmarkClassName?: string;
};

export const BrandMark = ({ className = "", decorative = false }: BrandMarkProps) => (
  <img
    src="/brand/sparkyai-logo.png"
    alt={decorative ? "" : "SparkyAI dragon logo"}
    className={className}
  />
);

const BrandLogo = ({
  className = "",
  markClassName = "h-11 w-auto",
  wordmarkClassName = "text-2xl",
}: BrandLogoProps) => (
  <span className={`inline-flex items-center gap-2.5 ${className}`}>
    <BrandMark className={markClassName} decorative />
    <span className={`${wordmarkClassName} font-bold tracking-tight text-sparky-maroon`}>
      Sparky<span className="text-sparky-gold">AI</span>
    </span>
  </span>
);

export default BrandLogo;
