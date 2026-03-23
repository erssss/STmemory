import React from 'react';

export default function DatabaseIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <ellipse
        cx="12"
        cy="5"
        rx="9"
        ry="3"
        fill="currentColor"
        fillOpacity="0.9"
      />
      <path
        d="M21 12C21 13.6569 16.9706 15 12 15C7.02944 15 3 13.6569 3 12"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        fill="none"
      />
      <path
        d="M21 5V19C21 20.6569 16.9706 22 12 22C7.02944 22 3 20.6569 3 19V5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        fill="none"
      />
      <ellipse
        cx="12"
        cy="12"
        rx="9"
        ry="3"
        fill="currentColor"
        fillOpacity="0.6"
      />
      <ellipse
        cx="12"
        cy="19"
        rx="9"
        ry="3"
        fill="currentColor"
        fillOpacity="0.9"
      />
    </svg>
  );
}


