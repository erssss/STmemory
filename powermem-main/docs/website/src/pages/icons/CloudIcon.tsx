import React from 'react';

export default function CloudIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M19.36 10.04C18.67 6.59 15.64 4 12 4C9.11 4 6.6 5.64 5.35 8.04C2.34 8.36 0 10.91 0 14C0 17.31 2.69 20 6 20H19C21.76 20 24 17.76 24 15C24 12.36 21.95 10.22 19.36 10.04ZM19 18H6C3.79 18 2 16.21 2 14C2 11.95 3.53 10.24 5.56 10.03L6.63 9.92L7.13 8.97C8.08 6.81 9.94 5.5 12 5.5C14.62 5.5 16.88 7.22 17.39 9.5L17.69 10.93L19.22 11.04C20.78 11.14 22 12.45 22 14C22 15.66 20.66 17 19 17V18Z"
        fill="currentColor"
      />
    </svg>
  );
}

