import { NullEngine } from '@babylonjs/core';
import { field, gpuStruct } from '../src/decorators';
import { ShadoActor } from '../src/extensions/ShadoActor';
import { ShadoInstanceContainer } from '../src/extensions/ShadoInstanceContainer/ShadoInstanceContainer';
import {
  ShadoFoliageActor,
  ShadoFoliageContainer,
} from '../src/extensions/ShadoFoliageContainer/ShadoFoliageContainer';

@gpuStruct({ name: 'CrossFamilyEntityActor' })
class CrossFamilyEntityActor extends ShadoActor {
  @field('u32') entityId!: number;
}

@gpuStruct({ name: 'CrossFamilyEntityContainer' })
class CrossFamilyEntityContainer extends ShadoInstanceContainer<CrossFamilyEntityActor> {}

describe('shader generation with coexisting container families', () => {
  it('keeps each family on its own actor header after another family initializes', async () => {
    const engine = new NullEngine();
    await CrossFamilyEntityContainer.initialize(engine, {
      wasm: false,
      extra: CrossFamilyEntityActor,
    });
    // The regression: this second initialize used to overwrite one shared
    // static, so any entity material compiled after it — lazily created
    // picking materials in the client — was generated against the foliage
    // actor header and failed WebGPU validation on every draw.
    await ShadoFoliageContainer.initialize(engine, {
      wasm: false,
      extra: ShadoFoliageActor,
    });

    const entities = new CrossFamilyEntityContainer(engine);
    const foliage = new ShadoFoliageContainer(engine);
    try {
      const entityGlsl = entities.generateGLSLPair().vs;
      const entityWgsl = entities.generateWGSLPair().vs;
      expect(entityGlsl).toContain('#include<CrossFamilyEntityActor>');
      expect(entityGlsl).not.toContain('ShadoFoliageActor');
      expect(entityWgsl).toContain('CrossFamilyEntityActor');
      expect(entityWgsl).not.toContain('ShadoFoliageActor');

      const foliageGlsl = foliage.generateGLSLPair().vs;
      expect(foliageGlsl).toContain('#include<ShadoFoliageActor>');
      expect(foliageGlsl).not.toContain('CrossFamilyEntityActor');
    } finally {
      entities.dispose();
      foliage.dispose();
      engine.dispose();
    }
  });
});
