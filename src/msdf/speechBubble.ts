/**
 * Speech bubbles.
 *
 * World-space dialogue rides the same MSDF layer as nameplates and floating combat
 * text: one font atlas, one shader, one glyph buffer. What a bubble adds over a
 * combat number is that it has to stay with whoever said it — an NPC who says a line
 * and walks away should take the line with them — so the pool stores a speaker rather
 * than a point, and the caller resolves that speaker's live position every frame.
 *
 * The reading budget is the whole design. A line that vanishes on a timer tuned for
 * damage numbers is unreadable prose, so lifetime scales with length and the text is
 * wrapped and clamped rather than shrunk: a bubble is a legible excerpt of what was
 * said, and chat keeps the full line. One bubble per speaker, always — a second line
 * replaces the first instead of stacking, because an NPC talking over itself reads as
 * a bug and costs the player the sentence they were mid-way through.
 *
 * Pure and clock-driven like `FloatingTextPool`: `advance` takes the current time and
 * returns what to draw, so the same code runs in the sandbox, in the game, and in a
 * test with no renderer.
 */

export type SpeechBubbleTone = "npc" | "player" | "emote";

export interface SpeechBubbleSpawn {
  /** Whatever the caller uses to find the speaker again — a spawn id, usually. */
  readonly speakerId: number | string;
  /** The spoken text, already stripped of any markup the chat layer understands. */
  readonly text: string;
  readonly tone?: SpeechBubbleTone;
}

/** Where a speaker is right now. Returning null retires the bubble. */
export interface SpeechBubbleAnchor {
  readonly x: number;
  readonly y: number;
  /** Top of the speaker, in the same space the nameplate anchor uses. */
  readonly z: number;
  /** False while the speaker is culled or hidden; the bubble waits rather than dying. */
  readonly visible?: boolean;
  /** Scales the lift so a large creature does not wear its words on its chin. */
  readonly scale?: number;
}

export interface SpeechBubbleInstance {
  id: string;
  text: string;
  x: number;
  y: number;
  z: number;
  visible: boolean;
  fontSize: number;
  color: string;
}

export interface SpeechBubbleStyle {
  readonly rgb: string;
  readonly fontSize: number;
}

/**
 * NPC speech is the warm default; a nearby player reads cooler so the two are
 * distinguishable at a glance in a crowd, and an emote is muted because it is
 * description rather than dialogue.
 */
export const DEFAULT_SPEECH_BUBBLE_STYLES: Readonly<
  Record<SpeechBubbleTone, SpeechBubbleStyle>
> = {
  npc: { rgb: "f4e4c1", fontSize: 13 },
  player: { rgb: "cfe0f5", fontSize: 13 },
  emote: { rgb: "b9a9c6", fontSize: 12 },
};

export interface SpeechBubbleOptions {
  /** Shortest a bubble ever lives, however terse the line. */
  readonly minLifetimeMs?: number;
  /** Longest a bubble ever lives, however long the line. */
  readonly maxLifetimeMs?: number;
  /** Reading budget. ~13 characters a second is unhurried but not sleepy. */
  readonly msPerCharacter?: number;
  /** Fraction of life at full opacity before the fade starts. */
  readonly holdFraction?: number;
  /** Fraction of life spent fading in, so a line arrives rather than blinks on. */
  readonly fadeInFraction?: number;
  /** Wrap width. Narrow keeps the bubble over the speaker instead of across the zone. */
  readonly maxCharactersPerLine?: number;
  /** Lines kept before the excerpt is elided. Chat still has the whole thing. */
  readonly maxLines?: number;
  /** World units between the speaker's nameplate anchor and the bubble's baseline. */
  readonly liftUnits?: number;
  /** Extra lift per line, so a tall bubble grows upward and never over the name. */
  readonly lineLiftUnits?: number;
  readonly styles?: Readonly<Record<SpeechBubbleTone, SpeechBubbleStyle>>;
  /** Most bubbles drawn at once. The newest speakers win. */
  readonly maxActive?: number;
}

const DEFAULTS = {
  minLifetimeMs: 3_000,
  maxLifetimeMs: 9_000,
  msPerCharacter: 75,
  holdFraction: 0.78,
  fadeInFraction: 0.06,
  maxCharactersPerLine: 32,
  maxLines: 3,
  liftUnits: 1.6,
  lineLiftUnits: 0.62,
  maxActive: 8,
} as const;

interface ActiveBubble {
  readonly id: string;
  readonly speakerId: number | string;
  readonly text: string;
  readonly lineCount: number;
  readonly tone: SpeechBubbleTone;
  readonly bornAtMs: number;
  readonly lifetimeMs: number;
}

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));

const toHexByte = (value: number): string =>
  Math.round(clamp01(value) * 255)
    .toString(16)
    .padStart(2, "0");

/**
 * Greedy word wrap with a hard split for anything longer than a line.
 *
 * A single unbreakable token — a URL, a name run together — would otherwise set the
 * bubble's width on its own and push the text off the speaker.
 */
export function wrapSpeechText(
  text: string,
  maxCharactersPerLine: number,
  maxLines: number,
): string {
  const words = text.replace(/\s+/g, " ").trim().split(" ");
  const lines: string[] = [];
  let line = "";

  const push = (value: string): void => {
    if (value) lines.push(value);
  };

  for (const word of words) {
    let remaining = word;
    // Hard-split an oversized token before the greedy pass ever sees it.
    while (remaining.length > maxCharactersPerLine) {
      push(line);
      line = "";
      lines.push(remaining.slice(0, maxCharactersPerLine));
      remaining = remaining.slice(maxCharactersPerLine);
    }
    if (!remaining) continue;
    if (!line) {
      line = remaining;
    } else if (line.length + 1 + remaining.length <= maxCharactersPerLine) {
      line = `${line} ${remaining}`;
    } else {
      push(line);
      line = remaining;
    }
  }
  push(line);

  if (lines.length <= maxLines) return lines.join("\n");

  // Elide rather than truncate mid-word: the ellipsis is the bubble telling the
  // player there is more of this line waiting in chat.
  const kept = lines.slice(0, maxLines);
  const last = kept[maxLines - 1] ?? "";
  kept[maxLines - 1] = last.length + 1 > maxCharactersPerLine
    ? `${last.slice(0, Math.max(0, maxCharactersPerLine - 1)).trimEnd()}…`
    : `${last.trimEnd()}…`;
  return kept.join("\n");
}

export class SpeechBubblePool {
  #active: ActiveBubble[] = [];
  #sequence = 0;

  constructor(private readonly options: SpeechBubbleOptions = {}) {}

  get activeCount(): number {
    return this.#active.length;
  }

  /**
   * Puts a line above a speaker, replacing whatever they were already saying.
   *
   * Returns the bubble id, or null if the text was empty once normalized — callers
   * pipe chat straight in and an all-markup line is not worth a bubble.
   */
  speak(spawn: SpeechBubbleSpawn, nowMs: number): string | null {
    const maxCharacters =
      this.options.maxCharactersPerLine ?? DEFAULTS.maxCharactersPerLine;
    const maxLines = this.options.maxLines ?? DEFAULTS.maxLines;
    const text = wrapSpeechText(spawn.text ?? "", maxCharacters, maxLines);
    if (!text) return null;

    // A speaker gets one bubble. Dropping the old one here rather than letting both
    // run keeps the newest sentence readable instead of overlapping the last.
    this.#active = this.#active.filter(
      (entry) => entry.speakerId !== spawn.speakerId,
    );

    const maxActive = this.options.maxActive ?? DEFAULTS.maxActive;
    // Drop the oldest rather than refusing the newest: in a busy plaza the line
    // that just landed is the one the player is looking for.
    while (this.#active.length >= maxActive) this.#active.shift();

    const minLifetime = this.options.minLifetimeMs ?? DEFAULTS.minLifetimeMs;
    const maxLifetime = this.options.maxLifetimeMs ?? DEFAULTS.maxLifetimeMs;
    const msPerCharacter =
      this.options.msPerCharacter ?? DEFAULTS.msPerCharacter;
    // Length drives life off the displayed excerpt, not the whole line: the player
    // is only ever asked to read what the bubble actually shows.
    const lifetimeMs = Math.min(
      maxLifetime,
      Math.max(minLifetime, text.replace(/\n/g, "").length * msPerCharacter),
    );

    this.#sequence += 1;
    const id = `speech:${spawn.speakerId}:${this.#sequence}`;
    this.#active.push({
      id,
      speakerId: spawn.speakerId,
      text,
      lineCount: text.split("\n").length,
      tone: spawn.tone ?? "npc",
      bornAtMs: nowMs,
      lifetimeMs,
    });
    return id;
  }

  /** Silences one speaker, e.g. when they despawn. */
  silence(speakerId: number | string): void {
    this.#active = this.#active.filter((entry) => entry.speakerId !== speakerId);
  }

  /** Drops everything, e.g. on zone change. */
  clear(): void {
    this.#active.length = 0;
  }

  /**
   * Retires expired bubbles and returns what should be drawn this frame.
   *
   * `resolveAnchor` is asked where each speaker is now. Returning null retires the
   * bubble — a speaker who left the world stops talking — while returning an anchor
   * with `visible: false` keeps it alive but undrawn, so a line survives the speaker
   * passing behind a building instead of being eaten by the culler.
   */
  advance(
    nowMs: number,
    resolveAnchor: (speakerId: number | string) => SpeechBubbleAnchor | null,
  ): SpeechBubbleInstance[] {
    const holdFraction = this.options.holdFraction ?? DEFAULTS.holdFraction;
    const fadeInFraction = this.options.fadeInFraction ?? DEFAULTS.fadeInFraction;
    const liftUnits = this.options.liftUnits ?? DEFAULTS.liftUnits;
    const lineLiftUnits = this.options.lineLiftUnits ?? DEFAULTS.lineLiftUnits;
    const styles = this.options.styles ?? DEFAULT_SPEECH_BUBBLE_STYLES;

    const drawn: SpeechBubbleInstance[] = [];
    const kept: ActiveBubble[] = [];

    for (const entry of this.#active) {
      const age = nowMs - entry.bornAtMs;
      if (age >= entry.lifetimeMs) continue;

      const anchor = resolveAnchor(entry.speakerId);
      if (!anchor) continue;
      kept.push(entry);

      const life = clamp01(age / entry.lifetimeMs);
      const style = styles[entry.tone] ?? DEFAULT_SPEECH_BUBBLE_STYLES.npc;
      const scale = anchor.scale ?? 1;

      // The bubble sits a fixed distance above the nameplate anchor and grows
      // upward with its line count, so the name underneath never moves and stays
      // where a player expects to click it.
      const z =
        anchor.z + (liftUnits + entry.lineCount * lineLiftUnits) * scale;

      const fadeIn = fadeInFraction > 0 ? clamp01(life / fadeInFraction) : 1;
      const fadeOut = life <= holdFraction
        ? 1
        : 1 - clamp01((life - holdFraction) / (1 - holdFraction));
      // Squared on the way out, matching combat text: the line stays readable
      // through most of the fade rather than spending half its life as a ghost.
      const alpha = fadeIn * fadeOut * fadeOut;

      drawn.push({
        id: entry.id,
        text: entry.text,
        x: anchor.x,
        y: anchor.y,
        z,
        visible: anchor.visible !== false && alpha > 0.02,
        fontSize: style.fontSize,
        color: `#${style.rgb}${toHexByte(alpha)}`,
      });
    }

    this.#active = kept;
    return drawn;
  }
}
