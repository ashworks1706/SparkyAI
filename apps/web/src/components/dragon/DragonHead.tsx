/** Head silhouette in reference units: neck, skull, forehead, brow, snout, nose, hooked chin, jaw and throat. */
const HEAD =
  "M -30 310 C 20 300 40 280 60 250 C 80 235 95 225 110 215 L 300 205 C 340 220 390 245 430 270 C 450 285 458 310 455 340 C 470 355 500 370 530 385 C 560 398 585 405 595 425 C 602 445 598 470 580 480 C 565 492 548 500 535 508 C 548 520 548 540 532 552 C 512 566 482 568 456 560 C 432 542 412 522 390 510 C 360 495 310 480 280 480 C 240 482 210 510 180 530 C 120 540 50 535 -30 530";

/** An eastern dragon head in profile facing right, drawn around its neck at the origin. */
const DragonHead = ({ className = "" }: { className?: string }) => (
  <g className={className} transform="translate(-30 -126) scale(0.3)" strokeLinejoin="miter" strokeMiterlimit="8">
    <g fill="var(--color-shu)" stroke="var(--color-kin)" strokeWidth="12">
      <path d="M 120 262 C 80 242 40 216 10 185 C 28 225 44 256 62 282 C 80 276 100 270 120 262 Z" />
      <path d="M 105 218 C 85 160 60 95 35 30 C 72 82 120 150 168 212 Z" />
      <path d="M 220 210 C 214 150 212 100 220 55 C 246 110 272 160 302 208 Z" />
    </g>
    <path d={`${HEAD} Z`} fill="var(--color-shu)" />
    <path d={HEAD} fill="none" stroke="var(--color-kin)" strokeWidth="12" />
    <g fill="none" stroke="var(--color-kin)" strokeWidth="9" strokeLinecap="butt">
      <path d="M 440 330 C 400 300 330 278 262 258" />
      <path d="M 548 492 C 510 476 440 462 360 460 C 320 460 290 455 270 440 C 258 428 255 410 262 394" />
      <path d="M 562 424 C 574 430 578 442 572 452" />
      <path d="M 165 528 C 152 500 152 470 166 448" />
      <path d="M 110 532 C 98 504 98 474 112 452" />
      <path d="M 55 532 C 44 506 44 480 58 460" />
    </g>
    <path
      d="M 300 332 C 322 306 372 306 396 346 C 366 366 320 362 300 332 Z"
      fill="var(--color-shu)"
      stroke="var(--color-kin)"
      strokeWidth="9"
    />
    <path d="M 350 316 C 342 330 342 346 350 360 C 358 346 358 330 350 316 Z" fill="var(--color-kin)" />
  </g>
);

export default DragonHead;
