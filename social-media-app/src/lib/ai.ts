export interface CaptionResult {
  caption: string;
  hashtags: string;
}

export async function generateCaption(
  extractedText: string,
  jobName: string,
  platforms: string[],
): Promise<CaptionResult> {
  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    // Fallback when OpenAI is not configured
    return {
      caption: `🎯 Today's predictions are live! Check out ${jobName} — expert picks powered by AI. Visit underdogs.bet`,
      hashtags:
        '#SportsPredictions #SportsBetting #Underdogs #AIPicks #WinningPicks',
    };
  }

  try {
    const { default: OpenAI } = await import('openai');
    const openai = new OpenAI({ apiKey });

    const platformList = platforms.join(', ');
    const textSnippet = extractedText.slice(0, 1_500);

    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content:
            'You are a sports betting social media manager for underdogs.bet, a professional ML-powered sports predictions platform. Write engaging, concise posts that excite fans.',
        },
        {
          role: 'user',
          content: `Generate a social media caption and hashtags for: ${platformList}

Content from our predictions page:
"""
${textSnippet}
"""

Rules:
- Caption: max 240 characters, engaging, include a call-to-action (e.g. "Check the link!")
- Hashtags: 6–8 tags relevant to sports betting/predictions
- Tone: confident, professional, exciting
- Always mention underdogs.bet

Respond as JSON: {"caption": "...", "hashtags": "..."}`,
        },
      ],
      response_format: { type: 'json_object' },
      max_tokens: 300,
    });

    const result = JSON.parse(
      response.choices[0].message.content ?? '{}',
    ) as Record<string, string>;

    return {
      caption: result.caption ?? '',
      hashtags: result.hashtags ?? '',
    };
  } catch (err) {
    console.error('[AI] Caption generation failed:', err);
    return {
      caption: `🎯 Today's ${jobName} predictions are live on underdogs.bet! Don't miss our AI-powered picks.`,
      hashtags: '#SportsPredictions #SportsBetting #Underdogs #AIPicks',
    };
  }
}
