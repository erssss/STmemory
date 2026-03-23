import React from 'react';

export default function ShieldIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M12 1L3 5V11C3 16.55 6.84 21.74 12 23C17.16 21.74 21 16.55 21 11V5L12 1ZM12 3.18L19 6.3V11C19 15.52 16.25 19.69 12 20.93C7.75 19.69 5 15.52 5 11V6.3L12 3.18ZM11 7V13L16 10L11 7Z"
        fill="currentColor"
      />
    </svg>
  );
}

