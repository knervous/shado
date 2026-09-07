import {
  SpeechBubblePool,
  wrapSpeechText,
  type SpeechBubbleAnchor,
} from '../src/msdf/speechBubble';

/**
 * Dialogue has to survive being read, which is a different bar from a damage number:
 * long enough to finish the sentence, wrapped narrow enough to stay over the speaker,
 * and stuck to a body that is walking away. Those are the properties asserted here.
 */

const alphaOf = (color: string): number => Number.parseInt(color.slice(-2), 16);

const anchorAt = (x: number, z = 6): SpeechBubbleAnchor => ({ x, y: 0, z });

describe('speech bubbles', () => {
  it('follows the speaker rather than the spot they spoke from', () => {
    const pool = new SpeechBubblePool();
    pool.speak({ speakerId: 7, text: 'Mind the carts.' }, 0);

    let x = 0;
    const at = (nowMs: number) => pool.advance(nowMs, () => anchorAt(x))[0]!;

    expect(at(0).x).toBe(0);
    x = 25;
    expect(at(500).x).toBe(25);
  });

  it('gives a long line more time to be read than a short one', () => {
    const shortPool = new SpeechBubblePool();
    const longPool = new SpeechBubblePool();
    shortPool.speak({ speakerId: 1, text: 'Hail.' }, 0);
    longPool.speak(
      {
        speakerId: 1,
        text: 'Past the gate the scrub is full of hares and adders and worse.',
      },
      0,
    );

    const alive = (pool: SpeechBubblePool, nowMs: number) =>
      pool.advance(nowMs, () => anchorAt(0)).length;

    // The floor keeps even a one-word line on screen long enough to notice.
    expect(alive(shortPool, 2_900)).toBe(1);
    expect(alive(shortPool, 4_000)).toBe(0);
    expect(alive(longPool, 4_000)).toBe(1);
  });

  it('holds full opacity while it is being read, then fades', () => {
    const pool = new SpeechBubblePool();
    pool.speak({ speakerId: 1, text: 'Shoulders back, traveler.' }, 0);
    const at = (nowMs: number) => pool.advance(nowMs, () => anchorAt(0))[0]!;

    // Fully faded in almost immediately; a line that blinks on is a line missed.
    expect(alphaOf(at(250).color)).toBe(255);
    expect(alphaOf(at(2_000).color)).toBe(255);
    const fading = at(2_700);
    expect(alphaOf(fading.color)).toBeLessThan(255);
    expect(alphaOf(fading.color)).toBeGreaterThan(0);
  });

  it('lets a speaker interrupt themselves instead of stacking', () => {
    const pool = new SpeechBubblePool();
    pool.speak({ speakerId: 4, text: 'Ask me about training.' }, 0);
    pool.speak({ speakerId: 4, text: 'Or a fight your size.' }, 100);

    const drawn = pool.advance(200, () => anchorAt(0));
    expect(drawn).toHaveLength(1);
    expect(drawn[0]!.text).toContain('fight your size');
  });

  it('retires a bubble whose speaker left the world, and waits out one merely hidden', () => {
    const pool = new SpeechBubblePool();
    pool.speak({ speakerId: 9, text: 'Wait here.' }, 0);

    // Behind a building: still alive, just not drawn.
    const hidden = pool.advance(100, () => ({ ...anchorAt(0), visible: false }))[0]!;
    expect(hidden.visible).toBe(false);
    expect(pool.activeCount).toBe(1);

    // Despawned: gone, and no longer holding a slot.
    expect(pool.advance(200, () => null)).toHaveLength(0);
    expect(pool.activeCount).toBe(0);
  });

  it('sits above the speaker and rises with its own line count', () => {
    const pool = new SpeechBubblePool({ maxCharactersPerLine: 12, maxLines: 3 });
    pool.speak({ speakerId: 1, text: 'one' }, 0);
    const oneLine = pool.advance(0, () => anchorAt(0, 6))[0]!;
    expect(oneLine.z).toBeGreaterThan(6);

    pool.speak({ speakerId: 2, text: 'one two three four' }, 0);
    const twoLines = pool.advance(0, () => anchorAt(0, 6))[1]!;
    expect(twoLines.z).toBeGreaterThan(oneLine.z);
  });

  it('drops the oldest speaker when a crowd talks at once', () => {
    const pool = new SpeechBubblePool({ maxActive: 2 });
    pool.speak({ speakerId: 1, text: 'first' }, 0);
    pool.speak({ speakerId: 2, text: 'second' }, 10);
    pool.speak({ speakerId: 3, text: 'third' }, 20);

    const drawn = pool.advance(30, () => anchorAt(0));
    expect(drawn.map((entry) => entry.text)).toEqual(['second', 'third']);
  });

  it('refuses a line that is empty once normalized', () => {
    const pool = new SpeechBubblePool();
    expect(pool.speak({ speakerId: 1, text: '   ' }, 0)).toBeNull();
    expect(pool.activeCount).toBe(0);
  });
});

describe('speech wrapping', () => {
  it('wraps on words and never exceeds the line budget', () => {
    const wrapped = wrapSpeechText(
      'Past the gate the scrub is full of hares and adders',
      20,
      5,
    );
    for (const line of wrapped.split('\n')) {
      expect(line.length).toBeLessThanOrEqual(20);
    }
    expect(wrapped).toContain('\n');
  });

  it('elides rather than running past the line budget', () => {
    const wrapped = wrapSpeechText(
      'one two three four five six seven eight nine ten eleven twelve',
      12,
      2,
    );
    expect(wrapped.split('\n')).toHaveLength(2);
    expect(wrapped.endsWith('…')).toBe(true);
  });

  it('hard-splits a token too long to fit, so one word cannot set the width', () => {
    const wrapped = wrapSpeechText('a'.repeat(30), 10, 4);
    for (const line of wrapped.split('\n')) {
      expect(line.length).toBeLessThanOrEqual(10);
    }
  });
});
