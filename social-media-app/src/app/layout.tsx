import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Social Media Generator — underdogs.bet',
  description: 'Automate daily social media posts from your sports prediction pages',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
