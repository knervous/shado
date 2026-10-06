import { type ShadoConfig, type PendingField } from '../decorators';
import { ShadoActor } from '../extensions/ShadoActor';
import { ShadoInstanceContainer } from '../extensions/ShadoInstanceContainer/ShadoInstanceContainer';
import { shadoPublish } from '../publish';
import { SHOWCASE_WEAPONS } from './EqShowcaseCatalog';
import { EQ_SHOWCASE_GLSL, EQ_SHOWCASE_WGSL } from './EqShowcaseShader';

export class EqShowcaseActor extends ShadoActor {
  static readonly shadoConfig: ShadoConfig = { name: 'EqShowcaseActor' };
  static readonly shadoFields: readonly PendingField[] = [
    { name: 'skinTint', type: 'vec4' },
    { name: 'chestTint', type: 'vec4' },
    { name: 'legTint', type: 'vec4' },
    { name: 'trimTint', type: 'vec4' },
    { name: 'armorClass', type: 'f32' },
    { name: 'weaponClass', type: 'f32' },
  ];
  skinTint!: Float32Array;
  chestTint!: Float32Array;
  legTint!: Float32Array;
  trimTint!: Float32Array;

  @shadoPublish({
    name: 'armor',
    label: 'Armor',
    group: 'Appearance',
    description: 'One complete Requiem material family across the whole character.',
    values: ['armorless', 'leather', 'chain', 'plate'],
  })
  armorClass!: number;

  @shadoPublish({
    name: 'mainHand',
    label: 'Main hand',
    group: 'Equipment',
    socket: 'r_point',
    description: 'Weapon attached to the EQ right-hand socket.',
    values: [
      { value: 'none', label: 'Unarmed' },
      ...SHOWCASE_WEAPONS.map((value, index) => ({
        value,
        label: `Weapon ${index + 1}`,
        description: `EQ right-hand model ${value}`,
      })),
    ],
  })
  weaponClass!: number;

  public override initialize() {
    super.initialize();
    this.skinTint = new Float32Array([1, 1, 1, 1]);
    this.chestTint = new Float32Array([1, 1, 1, 1]);
    this.legTint = new Float32Array([1, 1, 1, 1]);
    this.trimTint = new Float32Array([1, 1, 1, 1]);
    this.armorClass = 0;
    this.weaponClass = 0;
  }
}

export class EqShowcaseContainer extends ShadoInstanceContainer<EqShowcaseActor> {
  protected override getGLSLHooks() {
    return EQ_SHOWCASE_GLSL;
  }

  protected override getWGSLHooks() {
    return EQ_SHOWCASE_WGSL;
  }
}
