import { NextRequest, NextResponse } from 'next/server';
import { getDb } from '@/lib/db';
import { captureScreenshotAndText } from '@/lib/screenshot';
import { generateCaption } from '@/lib/ai';
import { postToAllPlatforms } from '@/lib/social';
import { v4 as uuidv4 } from 'uuid';

export const dynamic = 'force-dynamic';
// Allow up to 60 s for Puppeteer + OpenAI in production environments
export const maxDuration = 60;

interface DbJob {
  id: string;
  name: string;
  url: string;
  screenshot_selector: string;
  text_selector: string;
  platforms: string; // stored as JSON string
}

// ── POST /api/jobs/[id]/run ───────────────────────────────────────────────────
export async function POST(
  _request: NextRequest,
  { params }: { params: { id: string } },
) {
  const db = getDb();
  const job = db
    .prepare('SELECT * FROM social_jobs WHERE id = ?')
    .get(params.id) as DbJob | undefined;

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const platforms = JSON.parse(job.platforms) as string[];

  try {
    // ── 1. Screenshot + text extraction ────────────────────────────────────
    const capture = await captureScreenshotAndText(
      job.url,
      job.screenshot_selector,
      job.text_selector,
    );

    // ── 2. AI caption generation ────────────────────────────────────────────
    const { caption, hashtags } = await generateCaption(
      capture.extractedText ?? job.name,
      job.name,
      platforms,
    );

    // ── 3. Post to social platforms ─────────────────────────────────────────
    const platformResults = await postToAllPlatforms(
      platforms,
      caption,
      hashtags,
      capture.screenshotPath,
    );

    const allSuccess = platformResults.every((r) => r.success);
    const anySuccess = platformResults.some((r) => r.success);
    const status: string = allSuccess ? 'success' : anySuccess ? 'partial' : 'pending';

    // ── 4. Save to post history ─────────────────────────────────────────────
    const postId = uuidv4();
    const now = new Date().toISOString();

    db.prepare(`
      INSERT INTO post_history
        (id, job_id, job_name, caption, hashtags, screenshot_path, extracted_text,
         platforms, status, platform_results, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      postId,
      job.id,
      job.name,
      caption,
      hashtags,
      capture.screenshotPath,
      capture.extractedText,
      JSON.stringify(platforms),
      status,
      JSON.stringify(platformResults),
      now,
    );

    // ── 5. Update job's last_run ────────────────────────────────────────────
    db.prepare(
      'UPDATE social_jobs SET last_run = ?, last_status = ? WHERE id = ?',
    ).run(now, status, job.id);

    return NextResponse.json({
      success: true,
      postId,
      caption,
      hashtags,
      screenshotPath: capture.screenshotPath,
      extractedText: capture.extractedText,
      platformResults,
      status,
    });
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    console.error('[Run Job] Error:', errorMessage);

    db.prepare(
      'UPDATE social_jobs SET last_run = ?, last_status = ? WHERE id = ?',
    ).run(new Date().toISOString(), 'error', job.id);

    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}
