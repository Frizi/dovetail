use std::{
    collections::{hash_map::Entry, HashMap},
    iter::FromIterator,
    path::Path,
    str,
};

use wgpu::BindingType;

#[derive(Debug)]
pub struct Shader {
    filename: String,
    pub module: wgpu::ShaderModule,
    naga_module: naga::Module,
    info: naga::valid::ModuleInfo,
}

impl Shader {
    pub fn entry(&self, name: &str) -> ShaderEntryPoint<'_> {
        ShaderEntryPoint::by_name(self, name)
    }
}

pub struct ShaderEntryPoint<'a> {
    shader: &'a Shader,
    entry_point_index: usize,
}

pub struct ShaderSet<'a> {
    entry_points: Vec<ShaderEntryPoint<'a>>,
}

impl<'a> From<Vec<ShaderEntryPoint<'a>>> for ShaderSet<'a> {
    fn from(entry_points: Vec<ShaderEntryPoint<'a>>) -> Self {
        let mut stage_mask = wgpu::ShaderStages::empty();
        for ep in &entry_points {
            let stage = ep.stage();
            if stage_mask.intersects(stage) {
                panic!("Cannot use multiple entry points for single stage");
            }

            if (stage == wgpu::ShaderStages::COMPUTE && stage_mask != wgpu::ShaderStages::empty())
                || stage_mask.contains(wgpu::ShaderStages::COMPUTE)
            {
                panic!("Cannot mix compute and graphics shaders in single shader set.");
            }
            stage_mask |= stage;
        }

        if entry_points.is_empty() {
            panic!("Shader set cannot be empty.");
        }
        Self { entry_points }
    }
}

pub struct PipelineTemplate {
    layouts: Vec<wgpu::BindGroupLayout>,
    pipeline_layout: wgpu::PipelineLayout,
    binding_types: HashMap<String, ((u32, u32), BindingType)>,
}

pub struct BindGroupBuilder<'a> {
    template: &'a PipelineTemplate,
    group: u32,
    bindings: HashMap<u32, wgpu::BindingResource<'a>>,
}

impl PipelineTemplate {
    pub fn compute_pipeline(
        &self,
        device: &wgpu::Device,
        shader_set: &ShaderSet,
    ) -> wgpu::ComputePipeline {
        // TODO: assert that shader set is compatible
        assert_eq!(shader_set.entry_points.len(), 1);
        assert_eq!(
            shader_set.entry_points[0].stage(),
            wgpu::ShaderStages::COMPUTE
        );

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&self.pipeline_layout),
            module: &shader_set.entry_points[0].shader.module,
            entry_point: shader_set.entry_points[0].name(),
        })
    }

    pub fn graphics_pipeline(
        &self,
        device: &wgpu::Device,
        shader_set: &ShaderSet,
        buffers: &[wgpu::VertexBufferLayout<'_>],
        targets: &[wgpu::ColorTargetState],
        depth_stencil: Option<wgpu::DepthStencilState>,
    ) -> wgpu::RenderPipeline {
        // TODO: assert that shader set is compatible
        assert!(!shader_set.entry_points.is_empty());

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&self.pipeline_layout),
            vertex: shader_set.vertex_state(buffers),
            fragment: shader_set.fragment_state(targets),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }

    pub fn bind_group(&self, group: u32) -> BindGroupBuilder<'_> {
        BindGroupBuilder {
            template: self,
            group,
            bindings: Default::default(),
        }
    }
}

impl<'a> BindGroupBuilder<'a> {
    pub fn bind(&mut self, name: &str, resource: impl IntoResource<'a>) -> &mut Self {
        let (id, ty) = match self.template.binding_types.get(name) {
            Some(((g, b), ty)) if *g == self.group => (b, ty),
            _ => {
                // panic!("invalid binding name '{}'", name)
                return self;
            }
        };

        let resource = resource.into();
        match (ty, &resource) {
            (BindingType::Buffer { .. }, wgpu::BindingResource::Buffer(_))
            | (BindingType::Buffer { .. }, wgpu::BindingResource::BufferArray(_))
            | (BindingType::Sampler { .. }, wgpu::BindingResource::Sampler(_))
            | (
                BindingType::Texture { .. } | BindingType::StorageTexture { .. },
                wgpu::BindingResource::TextureView(_) | wgpu::BindingResource::TextureViewArray(_),
            ) => {}
            _ => panic!("incompatible binding type for '{}'", name),
        }
        self.bindings.insert(*id, resource);
        self
    }

    pub fn build(&self, device: &wgpu::Device) -> Option<wgpu::BindGroup> {
        if let Some(layout) = self.template.layouts.get(self.group as usize) {
            let entries = self
                .bindings
                .iter()
                .map(|(binding, resource)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: resource.clone(),
                })
                .collect::<Vec<_>>();

            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &entries,
            }))
        } else {
            None
        }
    }
}

pub trait IntoResource<'a> {
    fn into(self) -> wgpu::BindingResource<'a>;
}

impl<'a> IntoResource<'a> for wgpu::BindingResource<'a> {
    fn into(self) -> wgpu::BindingResource<'a> {
        self
    }
}

impl<'a> IntoResource<'a> for wgpu::BufferBinding<'a> {
    fn into(self) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Buffer(self)
    }
}

impl<'a> IntoResource<'a> for &'a wgpu::Buffer {
    fn into(self) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self,
            offset: 0,
            size: None,
        })
    }
}

impl<'a> IntoResource<'a> for &'a wgpu::TextureView {
    fn into(self) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::TextureView(self)
    }
}

impl<'a> IntoResource<'a> for &'a wgpu::Sampler {
    fn into(self) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Sampler(self)
    }
}

impl ShaderSet<'_> {
    pub fn pipeline_template(&self, device: &wgpu::Device) -> PipelineTemplate {
        let mut bindings: Vec<HashMap<u32, wgpu::BindGroupLayoutEntry>> = Vec::new();
        let mut push_constants: Vec<wgpu::PushConstantRange> = Vec::new();
        let mut binding_types = HashMap::new();
        for ep in &self.entry_points {
            let ep_info = ep.shader.info.get_entry_point(ep.entry_point_index);
            let stage = ep.stage();
            let module = &ep.shader.naga_module;
            for (var_handle, var) in module.global_variables.iter() {
                if let Some(ref binding) = var.binding {
                    let binding_use = ep_info[var_handle];
                    if !binding_use.is_empty() {
                        let ty = reflect_binding_type(module, var, binding_use);

                        if bindings.len() <= binding.group as usize {
                            bindings.resize_with(1 + binding.group as usize, Default::default);
                        }
                        let group_bindings = &mut bindings[binding.group as usize];
                        match group_bindings.entry(binding.binding) {
                            Entry::Occupied(mut entry) => {
                                let val = entry.get_mut();
                                val.visibility |= stage;
                                merge_bindings(&mut val.ty, ty);
                                binding_types.insert(
                                    var.name.clone().unwrap(),
                                    ((binding.group, binding.binding), val.ty),
                                );
                            }
                            Entry::Vacant(entry) => {
                                binding_types.insert(
                                    var.name.clone().unwrap(),
                                    ((binding.group, binding.binding), ty),
                                );
                                entry.insert(wgpu::BindGroupLayoutEntry {
                                    binding: binding.binding,
                                    visibility: stage,
                                    ty,
                                    count: None,
                                });
                            }
                        }
                    }
                } else if var.class == naga::StorageClass::PushConstant {
                    // TODO: figure out type size. For now, let's just assume single u32;
                    let range = 0..4;

                    if let Some(existing) = push_constants.iter_mut().find(|p| p.range == range) {
                        existing.stages |= stage;
                    } else {
                        push_constants.push(wgpu::PushConstantRange {
                            stages: stage,
                            range,
                        })
                    }
                }
            }
        }

        let layouts = bindings
            .into_iter()
            .map(|map| {
                let entries = map.into_iter().map(|(_, entry)| entry).collect::<Vec<_>>();
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &entries,
                })
            })
            .collect::<Vec<_>>();

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &layouts.iter().collect::<Vec<_>>(),
            push_constant_ranges: &push_constants,
        });

        PipelineTemplate {
            layouts,
            pipeline_layout,
            binding_types,
        }
    }

    pub fn vertex_state<'a>(
        &'a self,
        buffers: &'a [wgpu::VertexBufferLayout<'a>],
    ) -> wgpu::VertexState<'a> {
        let entry = self
            .entry_points
            .iter()
            .find(|e| e.stage() == wgpu::ShaderStages::VERTEX)
            .expect("Vertex shader required");

        wgpu::VertexState {
            module: &entry.shader.module,
            entry_point: entry.name(),
            buffers,
        }
    }

    pub fn fragment_state<'a>(
        &'a self,
        targets: &'a [wgpu::ColorTargetState],
    ) -> Option<wgpu::FragmentState<'a>> {
        let entry = self
            .entry_points
            .iter()
            .find(|e| e.stage() == wgpu::ShaderStages::FRAGMENT)?;

        Some(wgpu::FragmentState {
            module: &entry.shader.module,
            entry_point: entry.name(),
            targets,
        })
    }
}

fn merge_bindings(dst: &mut wgpu::BindingType, src: wgpu::BindingType) {
    match (dst, src) {
        (
            wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            },
            wgpu::BindingType::Buffer {
                ty: ty_src,
                has_dynamic_offset: has_dynamic_offset_src,
                min_binding_size: min_binding_size_src,
            },
        ) => {
            if ty != &ty_src {
                panic!("incompatible bindings: buffer type mismatch")
            }
            *has_dynamic_offset |= has_dynamic_offset_src;
            *min_binding_size = match (*min_binding_size, min_binding_size_src) {
                (None, size) => size,
                (size, None) => size,
                (Some(a), Some(b)) => Some(a.max(b)),
            }
        }
        (wgpu::BindingType::Sampler(binding), wgpu::BindingType::Sampler(binding_src)) => {
            match (binding, binding_src) {
                (wgpu::SamplerBindingType::Comparison, _) => {}
                (binding, wgpu::SamplerBindingType::Filtering) => {
                    *binding = wgpu::SamplerBindingType::Filtering
                }
                (binding, wgpu::SamplerBindingType::Comparison) => {
                    *binding = wgpu::SamplerBindingType::Comparison
                }
                (_, wgpu::SamplerBindingType::NonFiltering) => {}
            }
        }
        (
            wgpu::BindingType::Texture {
                view_dimension,
                sample_type,
                multisampled,
            },
            wgpu::BindingType::Texture {
                view_dimension: view_dimension_src,
                sample_type: sample_type_src,
                multisampled: multisampled_src,
            },
        ) => {
            if view_dimension != &view_dimension_src {
                panic!("incompatible bindings: texture view dimension mismatch");
            }
            if sample_type != &sample_type_src {
                panic!("incompatible bindings: texture sample type mismatch");
            }
            if multisampled != &multisampled_src {
                panic!("incompatible bindings: texture multisample state mismatch");
            }
        }
        (
            wgpu::BindingType::StorageTexture {
                access,
                format,
                view_dimension,
            },
            wgpu::BindingType::StorageTexture {
                access: access_src,
                format: format_src,
                view_dimension: view_dimension_src,
            },
        ) => {
            if format != &format_src {
                panic!("incompatible bindings: storage texture format mismatch");
            }
            if view_dimension != &view_dimension_src {
                panic!("incompatible bindings: storage texture dimension mismatch");
            }
            *access = match (*access, access_src) {
                (wgpu::StorageTextureAccess::ReadWrite, _)
                | (_, wgpu::StorageTextureAccess::ReadWrite) => {
                    wgpu::StorageTextureAccess::ReadWrite
                }
                (wgpu::StorageTextureAccess::ReadOnly, wgpu::StorageTextureAccess::WriteOnly)
                | (wgpu::StorageTextureAccess::WriteOnly, wgpu::StorageTextureAccess::ReadOnly) => {
                    wgpu::StorageTextureAccess::ReadWrite
                }
                _ => access_src,
            };
        }
        _ => panic!("incompatible bindings: binding type mismatch"),
    }
}

fn reflect_binding_type(
    module: &naga::Module,
    global: &naga::GlobalVariable,
    global_use: naga::valid::GlobalUse,
) -> wgpu::BindingType {
    match module.types[global.ty].inner {
        naga::TypeInner::Struct { .. } => wgpu::BindingType::Buffer {
            ty: match global.class {
                naga::StorageClass::Uniform => wgpu::BufferBindingType::Uniform,
                naga::StorageClass::Storage { access } => wgpu::BufferBindingType::Storage {
                    read_only: !access.contains(naga::StorageAccess::STORE)
                        && !global_use.contains(naga::valid::GlobalUse::WRITE),
                },
                _ => todo!(),
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        naga::TypeInner::Image {
            dim,
            arrayed,
            class: naga::ImageClass::Sampled { kind, multi },
        } => wgpu::BindingType::Texture {
            sample_type: match kind {
                naga::ScalarKind::Sint => wgpu::TextureSampleType::Sint,
                naga::ScalarKind::Uint => wgpu::TextureSampleType::Uint,
                naga::ScalarKind::Float => wgpu::TextureSampleType::Float {
                    filterable: false, // TODO: check if filtering is used in the shader
                },
                naga::ScalarKind::Bool => todo!(),
            },
            view_dimension: map_view_dimension(dim, arrayed),
            multisampled: multi,
        },
        naga::TypeInner::Image {
            dim,
            arrayed,
            class: naga::ImageClass::Storage { format, access },
        } => wgpu::BindingType::StorageTexture {
            access: {
                if access == naga::StorageAccess::all()
                    || global_use
                        .contains(naga::valid::GlobalUse::READ | naga::valid::GlobalUse::WRITE)
                {
                    wgpu::StorageTextureAccess::ReadWrite
                } else if access == naga::StorageAccess::STORE
                    || global_use.contains(naga::valid::GlobalUse::WRITE)
                {
                    wgpu::StorageTextureAccess::WriteOnly
                } else {
                    wgpu::StorageTextureAccess::ReadOnly
                }
            },
            format: map_format(format),
            view_dimension: map_view_dimension(dim, arrayed),
        },
        naga::TypeInner::Sampler { comparison: true } => {
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison)
        }
        naga::TypeInner::Sampler { comparison: false } => {
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering)
        }
        _ => todo!(),
    }
}

fn map_format(format: naga::StorageFormat) -> wgpu::TextureFormat {
    match format {
        naga::StorageFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
        naga::StorageFormat::R8Snorm => wgpu::TextureFormat::R8Snorm,
        naga::StorageFormat::R8Uint => wgpu::TextureFormat::R8Uint,
        naga::StorageFormat::R8Sint => wgpu::TextureFormat::R8Sint,
        naga::StorageFormat::R16Uint => wgpu::TextureFormat::R16Uint,
        naga::StorageFormat::R16Sint => wgpu::TextureFormat::R16Sint,
        naga::StorageFormat::R16Float => wgpu::TextureFormat::R16Float,
        naga::StorageFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
        naga::StorageFormat::Rg8Snorm => wgpu::TextureFormat::Rg8Snorm,
        naga::StorageFormat::Rg8Uint => wgpu::TextureFormat::Rg8Uint,
        naga::StorageFormat::Rg8Sint => wgpu::TextureFormat::Rg8Sint,
        naga::StorageFormat::R32Uint => wgpu::TextureFormat::R32Uint,
        naga::StorageFormat::R32Sint => wgpu::TextureFormat::R32Sint,
        naga::StorageFormat::R32Float => wgpu::TextureFormat::R32Float,
        naga::StorageFormat::Rg16Uint => wgpu::TextureFormat::Rg16Uint,
        naga::StorageFormat::Rg16Sint => wgpu::TextureFormat::Rg16Sint,
        naga::StorageFormat::Rg16Float => wgpu::TextureFormat::Rg16Float,
        naga::StorageFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        naga::StorageFormat::Rgba8Snorm => wgpu::TextureFormat::Rgba8Snorm,
        naga::StorageFormat::Rgba8Uint => wgpu::TextureFormat::Rgba8Uint,
        naga::StorageFormat::Rgba8Sint => wgpu::TextureFormat::Rgba8Sint,
        naga::StorageFormat::Rgb10a2Unorm => wgpu::TextureFormat::Rgb10a2Unorm,
        naga::StorageFormat::Rg11b10Float => wgpu::TextureFormat::Rg11b10Float,
        naga::StorageFormat::Rg32Uint => wgpu::TextureFormat::Rg32Uint,
        naga::StorageFormat::Rg32Sint => wgpu::TextureFormat::Rg32Sint,
        naga::StorageFormat::Rg32Float => wgpu::TextureFormat::Rg32Float,
        naga::StorageFormat::Rgba16Uint => wgpu::TextureFormat::Rgba16Uint,
        naga::StorageFormat::Rgba16Sint => wgpu::TextureFormat::Rgba16Sint,
        naga::StorageFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        naga::StorageFormat::Rgba32Uint => wgpu::TextureFormat::Rgba32Uint,
        naga::StorageFormat::Rgba32Sint => wgpu::TextureFormat::Rgba32Sint,
        naga::StorageFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
    }
}

fn map_view_dimension(dim: naga::ImageDimension, arrayed: bool) -> wgpu::TextureViewDimension {
    match (dim, arrayed) {
        (naga::ImageDimension::D1, false) => wgpu::TextureViewDimension::D1,
        (naga::ImageDimension::D2, false) => wgpu::TextureViewDimension::D2,
        (naga::ImageDimension::D2, true) => wgpu::TextureViewDimension::D2Array,
        (naga::ImageDimension::D3, false) => wgpu::TextureViewDimension::D3,
        (naga::ImageDimension::Cube, false) => wgpu::TextureViewDimension::Cube,
        (naga::ImageDimension::Cube, true) => wgpu::TextureViewDimension::CubeArray,
        _ => panic!(
            "invalid image dimension: {:?}, arrayed = {:?}",
            dim, arrayed
        ),
    }
}

impl<'a> FromIterator<ShaderEntryPoint<'a>> for ShaderSet<'a> {
    fn from_iter<T: IntoIterator<Item = ShaderEntryPoint<'a>>>(iter: T) -> Self {
        Self::from(iter.into_iter().collect::<Vec<_>>())
    }
}

impl<'a> ShaderEntryPoint<'a> {
    pub fn by_name(shader: &'a Shader, name: &str) -> Self {
        let entry_point_index = shader
            .naga_module
            .entry_points
            .iter()
            .position(|ep| ep.name == name)
            .unwrap_or_else(|| {
                panic!(
                    "Shader '{}' doesn't have entry point '{}'.",
                    shader.filename, name
                )
            });

        Self {
            shader,
            entry_point_index,
        }
    }

    fn name(&self) -> &str {
        &self.shader.naga_module.entry_points[self.entry_point_index].name
    }

    fn stage(&self) -> wgpu::ShaderStages {
        // self.shader.ent
        match self.shader.naga_module.entry_points[self.entry_point_index].stage {
            naga::ShaderStage::Vertex => wgpu::ShaderStages::VERTEX,
            naga::ShaderStage::Fragment => wgpu::ShaderStages::FRAGMENT,
            naga::ShaderStage::Compute => wgpu::ShaderStages::COMPUTE,
        }
    }
}

pub fn load_shader(device: &wgpu::Device, filename: impl AsRef<Path>) -> Shader {
    let filename_string = filename.as_ref().as_os_str().to_string_lossy().into_owned();
    let path = Path::new("shaders").join(filename);
    let shader_src =
        crate::pp::load_shader_preprocessed(&path).expect("failed to read shader file");

    let naga_module = naga::front::wgsl::parse_str(&shader_src).unwrap();
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::STRUCT_LAYOUTS,
        naga::valid::Capabilities::all(),
    );
    let info = validator.validate(&naga_module).unwrap();

    let shader_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    Shader {
        filename: filename_string,
        module: shader_module,
        naga_module,
        info,
    }
}
