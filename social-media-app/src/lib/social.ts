import type { PlatformResult } from '@/types';

// ─── Twitter / X ──────────────────────────────────────────────────────────────
export async function postToTwitter(
  caption: string,
  hashtags: string,
  _screenshotPath: string | null,
): Promise<PlatformResult> {
  const { TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET } =
    process.env;

  if (!TWITTER_API_KEY || !TWITTER_ACCESS_TOKEN) {
    return {
      platform: 'twitter',
      success: false,
      message:
        'Twitter API not configured — add TWITTER_API_KEY / TWITTER_ACCESS_TOKEN to .env.local',
    };
  }

  try {
    // Wire up: npm install twitter-api-v2
    // const { TwitterApi } = await import('twitter-api-v2');
    // const client = new TwitterApi({
    //   appKey: TWITTER_API_KEY,
    //   appSecret: TWITTER_API_SECRET!,
    //   accessToken: TWITTER_ACCESS_TOKEN,
    //   accessSecret: TWITTER_ACCESS_SECRET!,
    // });
    // const tweet = await client.v2.tweet(`${caption}\n\n${hashtags}`);
    // return { platform: 'twitter', success: true, message: 'Posted', postId: tweet.data.id };

    void TWITTER_API_SECRET;
    void TWITTER_ACCESS_SECRET;
    return {
      platform: 'twitter',
      success: false,
      message: 'Twitter posting stub — install twitter-api-v2 and uncomment the code in src/lib/social.ts',
    };
  } catch (err) {
    return { platform: 'twitter', success: false, message: String(err) };
  }
}

// ─── Instagram ────────────────────────────────────────────────────────────────
export async function postToInstagram(
  caption: string,
  hashtags: string,
  _screenshotPath: string | null,
): Promise<PlatformResult> {
  const { INSTAGRAM_ACCESS_TOKEN, INSTAGRAM_BUSINESS_ACCOUNT_ID } = process.env;

  if (!INSTAGRAM_ACCESS_TOKEN || !INSTAGRAM_BUSINESS_ACCOUNT_ID) {
    return {
      platform: 'instagram',
      success: false,
      message:
        'Instagram API not configured — add INSTAGRAM_ACCESS_TOKEN / INSTAGRAM_BUSINESS_ACCOUNT_ID to .env.local',
    };
  }

  try {
    // Wire up: use the Facebook Graph API for Instagram Business
    // See https://developers.facebook.com/docs/instagram-api/guides/content-publishing
    void caption;
    void hashtags;
    return {
      platform: 'instagram',
      success: false,
      message: 'Instagram posting stub — implement Facebook Graph API in src/lib/social.ts',
    };
  } catch (err) {
    return { platform: 'instagram', success: false, message: String(err) };
  }
}

// ─── Facebook ─────────────────────────────────────────────────────────────────
export async function postToFacebook(
  caption: string,
  hashtags: string,
  _screenshotPath: string | null,
): Promise<PlatformResult> {
  const { FACEBOOK_ACCESS_TOKEN, FACEBOOK_PAGE_ID } = process.env;

  if (!FACEBOOK_ACCESS_TOKEN || !FACEBOOK_PAGE_ID) {
    return {
      platform: 'facebook',
      success: false,
      message:
        'Facebook API not configured — add FACEBOOK_ACCESS_TOKEN / FACEBOOK_PAGE_ID to .env.local',
    };
  }

  try {
    // Wire up: POST to https://graph.facebook.com/{PAGE_ID}/feed
    void caption;
    void hashtags;
    return {
      platform: 'facebook',
      success: false,
      message: 'Facebook posting stub — implement Graph API POST in src/lib/social.ts',
    };
  } catch (err) {
    return { platform: 'facebook', success: false, message: String(err) };
  }
}

// ─── Dispatch to all platforms ────────────────────────────────────────────────
export async function postToAllPlatforms(
  platforms: string[],
  caption: string,
  hashtags: string,
  screenshotPath: string | null,
): Promise<PlatformResult[]> {
  const handlers: Record<string, () => Promise<PlatformResult>> = {
    twitter: () => postToTwitter(caption, hashtags, screenshotPath),
    instagram: () => postToInstagram(caption, hashtags, screenshotPath),
    facebook: () => postToFacebook(caption, hashtags, screenshotPath),
  };

  const settled = await Promise.allSettled(
    platforms.map((p) =>
      handlers[p]
        ? handlers[p]()
        : Promise.resolve<PlatformResult>({
            platform: p,
            success: false,
            message: `Unknown platform: ${p}`,
          }),
    ),
  );

  return settled.map((r, i) =>
    r.status === 'fulfilled'
      ? r.value
      : {
          platform: platforms[i],
          success: false,
          message: (r.reason as Error)?.message ?? 'Unknown error',
        },
  );
}
