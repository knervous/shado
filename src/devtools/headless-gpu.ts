/**
 * A real WebGPU device in Node, via Google Dawn.
 *
 * Babylon's WebGPU engine expects a browser, but it needs less of one than it
 * looks: `navigator.gpu`, a canvas that can hand back a WebGPU context, and a
 * couple of globals. Dawn supplies the adapter and this supplies the rest, so
 * the dev tools can drive the real engine — real materials, real shaders — with
 * no browser process.
 *
 * The native implementation is dawn-gpu/node-webgpu's `webgpu` package. It
 * ships Dawn for Node and exposes current optional features such as
 * `primitive-index` and `texture-component-swizzle`.
 */

export const TEXTURE_USAGE = {
  COPY_SRC: 0x01,
  COPY_DST: 0x02,
  TEXTURE_BINDING: 0x04,
  STORAGE_BINDING: 0x08,
  RENDER_ATTACHMENT: 0x10,
} as const;

export const BUFFER_USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
} as const;

/** Minimal GPUCanvasContext backed by a plain texture. */
class HeadlessCanvasContext {
  private device: any;
  private format = 'bgra8unorm';
  private usage: number = TEXTURE_USAGE.RENDER_ATTACHMENT;
  public texture: any = null;

  constructor(private readonly canvas: { width: number; height: number }) {}

  configure(config: any): void {
    this.device = config.device;
    this.format = config.format ?? 'bgra8unorm';
    this.usage = config.usage ?? TEXTURE_USAGE.RENDER_ATTACHMENT;
    this.texture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height],
      format: this.format,
      // TEXTURE_BINDING beyond what a swapchain needs, so a compute pass can
      // sample the drawn frame — that is what lets the yuv converter read a
      // surface directly instead of going through a CPU readback first.
      usage: this.usage | TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.COPY_SRC | TEXTURE_USAGE.TEXTURE_BINDING,
    });
  }
  unconfigure(): void { this.texture?.destroy?.(); this.texture = null; }
  getCurrentTexture(): any { return this.texture; }
  getConfiguration(): any { return { device: this.device, format: this.format, usage: this.usage }; }
}

/**
 * Dawn hands back `features` as a bare iterable and rejects the DOM
 * `addEventListener` signature; Babylon calls `.forEach` on the former and
 * registers `uncapturederror` on the latter. Note the listener call THROWS
 * rather than being absent, so optional-chaining is not sufficient.
 */
/** IEEE-754 half precision, for textures Dawn expects in float formats. */
function toHalf(value: number): number {
  const buffer = new Float32Array(1);
  const bits = new Uint32Array(buffer.buffer);
  buffer[0] = value;
  const x = bits[0]!;
  const sign = (x >>> 16) & 0x8000;
  let exponent = ((x >>> 23) & 0xff) - 112;
  let mantissa = x & 0x7fffff;
  if (exponent <= 0) return sign;
  if (exponent >= 0x1f) return sign | 0x7c00;
  mantissa >>= 13;
  return sign | (exponent << 10) | mantissa;
}

const BYTES_PER_PIXEL: Record<string, number> = {
  rgba8unorm: 4, 'rgba8unorm-srgb': 4, bgra8unorm: 4, 'bgra8unorm-srgb': 4,
  rgba16float: 8, rgba32float: 16,
};

type Descriptor = Record<string, unknown>;

function asDescriptor(value: unknown): Descriptor | undefined {
  return typeof value === 'object' && value !== null ? value as Descriptor : undefined;
}

function readComponent(value: unknown, key: string, index: number, fallback?: number): number {
  const descriptor = asDescriptor(value);
  const component = descriptor?.[key] ?? descriptor?.[String(index)];
  if (component === undefined && fallback !== undefined) return fallback;
  if (typeof component !== 'number' || !Number.isInteger(component) || component < 0) {
    throw new RangeError(`${key} must be a non-negative integer`);
  }
  return component;
}

function imageBytes(image: unknown): Uint8Array | null {
  if (ArrayBuffer.isView(image)) {
    return new Uint8Array(image.buffer, image.byteOffset, image.byteLength);
  }
  const data = asDescriptor(image)?.data;
  return ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : null;
}

function copyPackedRgba(
  bytes: Uint8Array,
  imageWidth: number,
  originX: number,
  originY: number,
  width: number,
  height: number,
  flipY: boolean,
): Uint8Array {
  const packed = new Uint8Array(width * height * 4);
  for (let destinationY = 0; destinationY < height; destinationY++) {
    const sourceY = originY + (flipY ? height - destinationY - 1 : destinationY);
    const sourceOffset = (sourceY * imageWidth + originX) * 4;
    packed.set(bytes.subarray(sourceOffset, sourceOffset + width * 4), destinationY * width * 4);
  }
  return packed;
}

function swizzleRgbaToBgra(bytes: Uint8Array): void {
  for (let offset = 0; offset < bytes.byteLength; offset += 4) {
    const red = bytes[offset]!;
    bytes[offset] = bytes[offset + 2]!;
    bytes[offset + 2] = red;
  }
}

/**
 * Converts 8-bit RGBA to a wider float format.
 *
 * A browser never needs this: Babylon uploads an ImageBitmap through
 * `copyExternalImageToTexture`, and the GPU converts on the way in. Dawn
 * rejects synthetic image sources, so uploads go through `writeTexture`
 * instead — which demands the data already be in the destination format.
 * Babylon's PBR BRDF lookup is the case that bites: an embedded 8-bit PNG
 * bound for an rgba16float texture, which fails as
 * "Required size for texture data layout (524288) exceeds the linear data
 * size (262144)".
 */
function widenRgba8(source: Uint8Array, pixels: number, format: string): ArrayBufferView {
  if (format === 'rgba16float') {
    const out = new Uint16Array(pixels * 4);
    for (let i = 0; i < pixels * 4; i++) out[i] = toHalf(source[i]! / 255);
    return out;
  }
  if (format === 'rgba32float') {
    const out = new Float32Array(pixels * 4);
    for (let i = 0; i < pixels * 4; i++) out[i] = source[i]! / 255;
    return out;
  }
  return source;
}

export function adaptQueue(queue: any): any {
  const writeTexture = (destination: any, data: any, layout: any, size: any): unknown => {
    const format: string | undefined = destination?.texture?.format;
    const needed = format ? BYTES_PER_PIXEL[format] : undefined;
    const width = size?.width ?? size?.[0] ?? 0;
    const height = size?.height ?? size?.[1] ?? 1;
    const view = ArrayBuffer.isView(data)
      ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
      : new Uint8Array(data);
    const supplied = width && height ? view.byteLength / (width * height) : 0;
    if (needed && needed !== 4 && Math.round(supplied) === 4) {
      const widened = widenRgba8(view, width * height, format!);
      return queue.writeTexture(destination, widened, { ...layout, bytesPerRow: width * needed, rowsPerImage: height }, size);
    }
    return queue.writeTexture(destination, data, layout, size);
  };

  /**
   * Dawn rejects synthetic sources for `copyExternalImageToTexture`, and our
   * `createImageBitmap` shim only produces synthetic ones. Babylon's core
   * engine sidesteps this by branching on `byteLength` and calling
   * `writeTexture` itself; Babylon Lite calls the external-image path
   * unconditionally, so the translation has to live here instead.
   */
  const copyExternalImageToTexture = (source: unknown, destination: unknown, copySize: unknown): unknown => {
    const sourceDescriptor = asDescriptor(source);
    const image = sourceDescriptor?.source ?? source;
    const bytes = imageBytes(image);
    if (!bytes) return queue.copyExternalImageToTexture(source, destination, copySize);

    const imageDescriptor = asDescriptor(image);
    const imageWidth = readComponent(imageDescriptor, 'width', 0);
    const imageHeight = readComponent(imageDescriptor, 'height', 1);
    if (bytes.byteLength < imageWidth * imageHeight * 4) {
      throw new RangeError('External image data is smaller than its RGBA dimensions');
    }

    const sourceOrigin = sourceDescriptor?.origin;
    const originX = readComponent(sourceOrigin, 'x', 0, 0);
    const originY = readComponent(sourceOrigin, 'y', 1, 0);
    const width = readComponent(copySize, 'width', 0);
    const height = readComponent(copySize, 'height', 1, 1);
    const depthOrArrayLayers = readComponent(copySize, 'depthOrArrayLayers', 2, 1);
    if (depthOrArrayLayers > 1) {
      throw new RangeError('copyExternalImageToTexture requires depthOrArrayLayers to be at most 1');
    }
    if (originX + width > imageWidth || originY + height > imageHeight) {
      throw new RangeError('External image copy exceeds the source dimensions');
    }

    const destinationDescriptor = asDescriptor(destination);
    if (destinationDescriptor?.premultipliedAlpha) {
      throw new TypeError('Headless external-image uploads do not support premultipliedAlpha: true');
    }
    const colorSpace = destinationDescriptor?.colorSpace;
    if (colorSpace !== undefined && colorSpace !== 'srgb') {
      throw new TypeError(`Headless external-image uploads do not support colorSpace: ${String(colorSpace)}`);
    }

    const texture = destinationDescriptor?.texture;
    const format = asDescriptor(texture)?.format;
    const bytesPerPixel = typeof format === 'string' ? BYTES_PER_PIXEL[format] ?? 4 : 4;
    const needsBgraSwizzle = format === 'bgra8unorm' || format === 'bgra8unorm-srgb';
    const flipY = Boolean(sourceDescriptor?.flipY);
    let uploadBytes: ArrayBufferView = bytes;
    let offset = (originY * imageWidth + originX) * 4;
    let bytesPerRow = imageWidth * 4;
    let rowsPerImage = imageHeight;

    if (depthOrArrayLayers === 0) return;

    if (flipY || bytesPerPixel !== 4 || needsBgraSwizzle) {
      const packed = copyPackedRgba(bytes, imageWidth, originX, originY, width, height, flipY);
      if (needsBgraSwizzle) swizzleRgbaToBgra(packed);
      uploadBytes = bytesPerPixel === 4 ? packed : widenRgba8(packed, width * height, String(format));
      offset = 0;
      bytesPerRow = width * bytesPerPixel;
      rowsPerImage = height;
    }

    const writeDestination: Descriptor = {
      texture,
      mipLevel: destinationDescriptor?.mipLevel ?? 0,
      origin: destinationDescriptor?.origin ?? { x: 0, y: 0, z: 0 },
    };
    if (destinationDescriptor?.aspect !== undefined) writeDestination.aspect = destinationDescriptor.aspect;

    return queue.writeTexture(
      writeDestination,
      uploadBytes,
      { offset, bytesPerRow, rowsPerImage },
      { width, height, depthOrArrayLayers: 1 },
    );
  };

  return new Proxy(queue, {
    get(object, key) {
      if (key === 'writeTexture') return writeTexture;
      if (key === 'copyExternalImageToTexture') return copyExternalImageToTexture;
      const value = object[key];
      return typeof value === 'function' ? value.bind(object) : value;
    },
  });
}

function adaptDawnObject(target: any): any {
  return new Proxy(target, {
    get(object, key) {
      if (key === 'features') return new Set(object.features);
      if (key === 'queue') return adaptQueue(object.queue);
      if (key === 'addEventListener' || key === 'removeEventListener') return () => {};
      const value = object[key];
      if (typeof value !== 'function') return value;
      if (key === 'requestDevice') {
        return async (...args: unknown[]) => adaptDawnObject(await value.apply(object, args));
      }
      return value.bind(object);
    },
  });
}

export interface HeadlessCanvas {
  width: number;
  height: number;
  _context: HeadlessCanvasContext;
  getContext(kind: string): unknown;
}

export function createHeadlessCanvas(width: number, height: number): HeadlessCanvas {
  const canvas: any = {
    width, height, clientWidth: width, clientHeight: height, style: {},
    getContext(kind: string) { return kind === 'webgpu' ? this._context : null; },
    addEventListener() {}, removeEventListener() {},
    getBoundingClientRect: () => ({ width, height, left: 0, top: 0, right: width, bottom: height }),
    setAttribute() {}, removeAttribute() {}, focus() {},
  };
  canvas._context = new HeadlessCanvasContext(canvas);
  return canvas as HeadlessCanvas;
}

export interface HeadlessGpu {
  gpu: any;
  /** Must run before the process exits — an undestroyed Dawn instance hangs Node. */
  dispose(): void;
}

export interface DecodedImage {
  width: number;
  height: number;
  /** Tightly packed RGBA8. */
  data: Uint8Array;
}

/** Decodes PNG/JPEG/WebP bytes to RGBA. `sharp` satisfies this. */
export type ImageDecoder = (bytes: Uint8Array, mimeType: string) => Promise<DecodedImage>;

function assertSupportedImageBitmapOptions(options: unknown): void {
  if (options === undefined) return;
  const descriptor = asDescriptor(options);
  if (!descriptor) throw new TypeError('createImageBitmap options must be an object');

  const unsupported = [
    descriptor.imageOrientation !== undefined && descriptor.imageOrientation !== 'from-image' ? 'imageOrientation' : null,
    descriptor.premultiplyAlpha !== undefined
      && descriptor.premultiplyAlpha !== 'default'
      && descriptor.premultiplyAlpha !== 'none' ? 'premultiplyAlpha' : null,
    descriptor.colorSpaceConversion !== undefined
      && descriptor.colorSpaceConversion !== 'default'
      && descriptor.colorSpaceConversion !== 'none' ? 'colorSpaceConversion' : null,
    descriptor.resizeWidth !== undefined ? 'resizeWidth' : null,
    descriptor.resizeHeight !== undefined ? 'resizeHeight' : null,
    descriptor.resizeQuality !== undefined ? 'resizeQuality' : null,
  ].find((name): name is string => name !== null);
  if (unsupported) {
    throw new TypeError(`createImageBitmap shim does not support the ${unsupported} option`);
  }
}

/**
 * Lets Babylon's real texture pipeline run in Node.
 *
 * Babylon decodes images through `createImageBitmap`, then uploads whatever it
 * gets back. Its WebGPU upload branches on one thing:
 *
 *     if (imageBitmap.byteLength !== undefined) -> queue.writeTexture(...)
 *     else                                     -> queue.copyExternalImageToTexture(...)
 *
 * Dawn rejects synthetic sources for `copyExternalImageToTexture` but accepts
 * `writeTexture` with raw bytes. So this returns a Uint8Array of RGBA with the
 * dimensions attached: it has a byteLength, so Babylon takes the raw-data path
 * on its own, and no part of the engine or the device needs patching.
 */
export function installImageDecoder(decode: ImageDecoder): void {
  if (typeof (globalThis as any).createImageBitmap === 'function') return;
  (globalThis as any).createImageBitmap = async (source: unknown, ...arguments_: unknown[]): Promise<unknown> => {
    if (arguments_.length >= 4) {
      throw new TypeError('createImageBitmap shim does not support the crop overload');
    }
    if (arguments_.length > 1) {
      throw new TypeError('Invalid createImageBitmap arguments');
    }
    assertSupportedImageBitmapOptions(arguments_[0]);

    let bytes: Uint8Array;
    let mimeType = 'image/png';
    if (source instanceof Uint8Array) bytes = source;
    else if (typeof Blob !== 'undefined' && source instanceof Blob) {
      bytes = new Uint8Array(await source.arrayBuffer());
      mimeType = source.type || mimeType;
    } else if (source instanceof ArrayBuffer) bytes = new Uint8Array(source);
    else if (ArrayBuffer.isView(source)) bytes = new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
    else throw new Error(`createImageBitmap shim cannot read a ${Object.prototype.toString.call(source)}`);
    const image = await decode(bytes, mimeType);
    const view = new Uint8Array(image.data) as Uint8Array & { width: number; height: number; close(): void };
    view.width = image.width;
    view.height = image.height;
    view.close = () => {};
    return view;
  };
}

/**
 * Babylon's scene loader reaches for XMLHttpRequest on some paths even when it
 * was handed bytes, so Node needs one to exist. `preprocess/models` installs a
 * fuller version for the bake pipeline; this is the small equivalent, kept
 * local so the dev tools do not drag the model pipeline in behind them.
 */
function installXmlHttpRequestShim(): void {
  if (typeof (globalThis as any).XMLHttpRequest !== 'undefined') return;
  class FetchXhr {
    // Babylon reads XMLHttpRequest.DONE off the constructor and waits on
    // 'readystatechange' / 'loadend' listeners, not on onload.
    static readonly UNSENT = 0;
    static readonly OPENED = 1;
    static readonly HEADERS_RECEIVED = 2;
    static readonly LOADING = 3;
    static readonly DONE = 4;
    public readyState = 0;
    public status = 0;
    public statusText = '';
    public response: unknown = null;
    public responseText = '';
    public responseType = '';
    public responseURL = '';
    public onreadystatechange: (() => void) | null = null;
    public onload: (() => void) | null = null;
    public onerror: ((error?: unknown) => void) | null = null;
    private listeners = new Map<string, Array<(event?: unknown) => void>>();
    private method = 'GET';
    private url = '';
    private contentType: string | null = null;

    open(method: string, url: string): void { this.method = method; this.url = url; this.readyState = 1; }
    setRequestHeader(): void {}
    getAllResponseHeaders(): string { return this.contentType ? `content-type: ${this.contentType}` : ''; }
    getResponseHeader(name: string): string | null {
      return name.toLowerCase() === 'content-type' ? this.contentType : null;
    }
    abort(): void {}
    addEventListener(type: string, handler: (event?: unknown) => void): void {
      this.listeners.set(type, [...(this.listeners.get(type) ?? []), handler]);
    }
    removeEventListener(): void {}
    private dispatch(type: string): void {
      for (const handler of this.listeners.get(type) ?? []) handler({ target: this });
    }
    /**
     * Settles the request the way Babylon's RequestFile expects: it attaches
     * 'readystatechange' and 'loadend' listeners and inspects readyState and
     * status from there. Firing only onload leaves it waiting forever.
     */
    private finish(kind: 'load' | 'error'): void {
      this.readyState = 4;
      this.onreadystatechange?.();
      this.dispatch('readystatechange');
      this.dispatch(kind);
      if (kind === 'load') this.onload?.(); else this.onerror?.();
      this.dispatch('loadend');
    }
    send(): void {
      void (async () => {
        try {
          // Babylon wraps embedded images in a Blob and hands the loader a
          // blob: URL. Node's fetch cannot resolve those — resolveObjectURL is
          // the only way back to the bytes, and without this the loader waits
          // forever on an image that never arrives.
          let buffer: ArrayBuffer;
          if (this.url.startsWith('blob:')) {
            const { resolveObjectURL } = await import('node:buffer');
            const blob = resolveObjectURL(this.url);
            if (!blob) throw new Error(`Unresolvable object URL ${this.url}`);
            buffer = await blob.arrayBuffer();
            this.contentType = blob.type || null;
            this.status = 200;
            this.statusText = 'OK';
          } else {
            const response = await fetch(this.url, { method: this.method });
            buffer = await response.arrayBuffer();
            this.status = response.status;
            this.statusText = response.statusText;
            this.contentType = response.headers.get('content-type');
          }
          this.responseURL = this.url;
          this.response = this.responseType === 'text' ? new TextDecoder().decode(buffer) : buffer;
          if (this.responseType === '' || this.responseType === 'text') this.responseText = new TextDecoder().decode(buffer);
          this.finish('load');
        } catch {
          this.status = 0;
          this.finish('error');
        }
      })();
    }
  }
  (globalThis as any).XMLHttpRequest = FetchXhr;
}

/**
 * WebGPU's constant namespaces, which the spec exposes as globals.
 *
 * `webgpu` provides these through its `globals` export. The numeric fallback
 * keeps the shim explicit and protects against an incomplete native export.
 */
const WEBGPU_CONSTANTS: Record<string, Record<string, number>> = {
  GPUBufferUsage: {
    MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
    INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
    INDIRECT: 0x0100, QUERY_RESOLVE: 0x0200,
  },
  GPUTextureUsage: {
    COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04,
    STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10,
  },
  GPUShaderStage: { VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 },
  GPUMapMode: { READ: 0x1, WRITE: 0x2 },
  GPUColorWrite: { RED: 0x1, GREEN: 0x2, BLUE: 0x4, ALPHA: 0x8, ALL: 0xf },
};

/** Installs `navigator.gpu` and the globals Babylon's WebGPU engine reads. */
export async function installHeadlessWebGpu(): Promise<HeadlessGpu> {
  const dawn = await import('webgpu');
  const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  let instance: any = dawn.create([]);
  // node-webgpu expects its WebGPU constants (GPUBufferUsage, GPUMapMode, ...)
  // to be installed globally.
  if (dawn.globals) for (const [key, value] of Object.entries(dawn.globals)) (globalThis as any)[key] ??= value;
  const gpu = {
    requestAdapter: async (options?: unknown) => {
      if (!instance) throw new Error('Headless WebGPU has been disposed');
      const adapter = await instance.requestAdapter(options);
      return adapter ? adaptDawnObject(adapter) : adapter;
    },
    getPreferredCanvasFormat: () => 'bgra8unorm',
    wgslLanguageFeatures: new Set<string>(instance.wgslLanguageFeatures ?? []),
  };
  // Node 24 exposes `navigator` as a getter-only global, so redefine it.
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: { ...globalThis.navigator, gpu, userAgent: 'node', language: 'en', onLine: true },
  });
  // Only fill what the binding left out, so a binding that provides its own
  // (and may know better) keeps them.
  for (const [name, values] of Object.entries(WEBGPU_CONSTANTS)) {
    (globalThis as any)[name] ??= Object.freeze({ ...values });
  }
  installXmlHttpRequestShim();
  (globalThis as any).self ??= globalThis;
  (globalThis as any).requestAnimationFrame ??= (callback: (t: number) => void) => setTimeout(() => callback(Date.now()), 0);
  (globalThis as any).cancelAnimationFrame ??= (id: any) => clearTimeout(id);
  return {
    gpu,
    dispose: () => {
      if (!instance) return;
      if ((globalThis as any).navigator?.gpu === gpu) {
        if (previousNavigator) Object.defineProperty(globalThis, 'navigator', previousNavigator);
        else delete (globalThis as any).navigator;
      }
      instance = null;
    },
  };
}
