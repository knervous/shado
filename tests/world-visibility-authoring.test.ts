import { createShadoWorldAuthoring, validateShadoWorldAuthoring } from '../src/world';

/**
 * A zone's disocclusion sources live in its authoring document; declaring
 * them makes the disocclusion PVS a mandatory stage of the zone's promotion.
 * A malformed block must fail validation (at save and at promotion), not
 * reach the baker.
 */
describe('visibility.disocclusion authoring', () => {
  const source = { id: 'court', min: [-1, 4, -1] as [number, number, number], max: [1, 6, 1] as [number, number, number], near: 8, far: 1300, sideUp: 0.6 };
  const withBlock = (disocclusion: unknown) => ({ ...createShadoWorldAuthoring('talioscrypts1'), visibility: { disocclusion } });

  it('accepts sources, numeric settings, review poses and a note', () => {
    const document = validateShadoWorldAuthoring(
      withBlock({
        note: 'arrival court at eye height',
        sources: [source],
        settings: { tileSize: 4 },
        poses: [{ name: 'court', at: [0, 5.1, 0], look: [0, 5.1, 40] }],
      })
    );
    expect(document.visibility?.disocclusion?.sources[0]?.id).toBe('court');
  });

  it('a zone without the block is valid and bakes nothing', () => {
    expect(validateShadoWorldAuthoring(createShadoWorldAuthoring('talioscrypts1')).visibility).toBeUndefined();
  });

  it.each([
    ['no sources', { sources: [] }, /at least one source/],
    ['duplicate ids', { sources: [source, source] }, /unique IDs/],
    ['empty box', { sources: [{ ...source, max: [1, 4, 1] }] }, /box is empty/],
    ['far before near', { sources: [{ ...source, far: 4 }] }, /far must exceed near/],
    ['non-positive setting', { sources: [source], settings: { tileSize: 0 } }, /tileSize/],
    ['pose without a name', { sources: [source], poses: [{ name: '', at: [0, 0, 0], look: [0, 0, 1] }] }, /names/],
  ])('refuses %s', (_label, block, message) => {
    expect(() => validateShadoWorldAuthoring(withBlock(block))).toThrow(message);
  });
});
