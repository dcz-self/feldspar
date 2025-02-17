use crate::prelude::{ArrayMaterial, MeshMaterial};

use bevy::{
    asset::prelude::*,
    ecs::prelude::*,
    render::{prelude::*, texture::AddressMode},
};

/// The set of assets needed by the renderer. This resource must exist and all asset handles must have finished loading before
/// entering the update state configured by `VoxelWorldPlugin::new`.
///
/// In practice, this is a single texture and only used to define parameters to `spawn_aray_material`.
#[derive(Clone)]
pub struct VoxelRenderAssets {
    /// A 2D texture containing vertically stacked images of the same size. Each image corresponds to one voxel type.
    pub mesh_base_color: Handle<Texture>,
    pub image_count: u32,
}

impl From<VoxelRenderAssets> for ArrayMaterial {
    fn from(texture: VoxelRenderAssets) -> Self {
        assert_eq!(texture.image_count, 4);
        ArrayMaterial {
            base_color_texture: Some(texture.mesh_base_color),
            ..Default::default()
        }
    }
}

/// Convenience method: creates an ArrayMaterial and reformats its texture
/// to be compatible with `renderer::create_voxel_mesh_bundle`.
pub fn spawn_array_material<T: From<Handle<ArrayMaterial>> + Send + Sync + 'static>(
    assets: &VoxelRenderAssets,
    mut commands: Commands,
    mut array_materials: ResMut<Assets<ArrayMaterial>>,
    mut textures: ResMut<Assets<Texture>>,
) {
    let texture = textures
        .get_mut(&assets.mesh_base_color)
        .expect("mesh_base_color texture does not exist");

    handle_loaded_array_texture(texture, assets.image_count);

    let mut material = ArrayMaterial::from(assets.clone());
    material.roughness = 0.8;
    material.reflectance = 0.2;
    commands.insert_resource(T::from(array_materials.add(material)));
}

fn handle_loaded_array_texture(texture: &mut Texture, num_layers: u32) {
    if texture.size.depth != num_layers {
        texture.reinterpret_stacked_2d_as_array(num_layers);
    }
    texture.sampler.address_mode_u = AddressMode::Repeat;
    texture.sampler.address_mode_v = AddressMode::Repeat;
}
