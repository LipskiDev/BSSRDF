#include <cgv/base/register.h>
#include <filesystem>
#include <cgv/gui/provider.h>
#include <cgv/gui/trigger.h>
#include <cgv/math/ftransform.h>
#include <cgv/render/attribute_array_binding.h>
#include <cgv/render/drawable.h>
#include <cgv/render/frame_buffer.h>
#include <cgv/render/shader_program.h>
#include <cgv/render/texture.h>
#include <cgv/render/vertex_buffer.h>
#include <cgv/render/shader_library.h>
#include <cgv_gl/gl/mesh_render_info.h>

#include <fstream>
#include <sstream>
#include <random>
#include <limits>
#include <algorithm>

#include "cgv/math/fvec.h"
#include <cgv_gl/gl/gl.h>
#include <plugins/cmf_tt_gl_font/tt_gl_font_server.h>

#define FB_MAX_RESOLUTION 2048
#define BLUR_PASSES 5
#define DOM_LAYER_COUNT 16

using namespace cgv::render;

class bssrdf
	: public cgv::base::node,
	public cgv::gui::provider,
	public cgv::render::drawable
{
protected:
	struct render_context {
		context* ctx = nullptr;
		cgv::mat4 view_matrix;
		cgv::mat4 light_matrix;

		void store_view() {
			view_matrix = ctx->get_modelview_matrix();
		}

		cgv::mat3 get_normal_matrix(const cgv::mat4& M) {
			cgv::math::fmat<float, 3, 3> NM;
			NM(0, 0) = M(0, 0); NM(0, 1) = M(0, 1); NM(0, 2) = M(0, 2);
			NM(1, 0) = M(1, 0); NM(1, 1) = M(1, 1); NM(1, 2) = M(1, 2);
			NM(2, 0) = M(2, 0); NM(2, 1) = M(2, 1); NM(2, 2) = M(2, 2);
			NM.transpose();
			NM = inv(NM);
			return NM;
		}
	};

	template<typename T>
	struct scene_object {
		cgv::vec3 position = cgv::vec3(0.0f);
		cgv::vec3 rotation = cgv::vec3(0.0f);
		cgv::vec3 scale = cgv::vec3(1.0f);
		cgv::vec2 uv_scale = cgv::vec2(1.0f);

		cgv::mat4 transformation_matrix;

		std::vector<T> verts;
		cgv::render::vertex_buffer vb;
		cgv::render::attribute_array_binding va;

		texture albedo_tex, metallic_tex, roughness_tex, normal_tex;

		scene_object() {
			transformation_matrix.identity();
		}

		void compute_transformation() {
			transformation_matrix =
				cgv::math::translate4(position) *
				cgv::math::rotate4(rotation) *
				cgv::math::scale4(scale);
		}

		void set_common_uniforms(render_context& rctx, shader_program& shader) {
			context& ctx = *rctx.ctx;
			shader.set_uniform(ctx, "u_model_matrix", transformation_matrix);
			shader.set_uniform(ctx, "u_model_normal_matrix", rctx.get_normal_matrix(transformation_matrix));
			shader.set_uniform(ctx, "u_lightSpaceMatrix", rctx.light_matrix * transformation_matrix);
			shader.set_uniform(ctx, "u_uv_scale", uv_scale);
		}

		void set_modelview(render_context& rctx) {
			context& ctx = *rctx.ctx;
			ctx.set_modelview_matrix(rctx.view_matrix * transformation_matrix);
		}

		void draw_bound(render_context& rctx, shader_program& shader) {
			context& ctx = *rctx.ctx;
			set_common_uniforms(rctx, shader);
			set_modelview(rctx);

			ctx.push_modelview_matrix();
			va.enable(ctx);
			glDrawArrays(GL_TRIANGLES, 0, (GLsizei)verts.size());
			va.disable(ctx);
			ctx.pop_modelview_matrix();
		}

		void draw_depth(render_context& rctx, shader_program& shader) {
			context& ctx = *rctx.ctx;
			shader.set_uniform(ctx, "u_lightSpaceMatrix", rctx.light_matrix * transformation_matrix);
			va.enable(ctx);
			glDrawArrays(GL_TRIANGLES, 0, (GLsizei)verts.size());
			va.disable(ctx);
		}
	};

	struct vertex {
		cgv::vec3 pos;
		cgv::vec3 normal;
		cgv::vec2 tcoord;
	};

	struct strand_vertex {
		cgv::vec3 center;
		cgv::vec3 tan;
		cgv::vec2 rootUV;
		float VAlong;
		float side;
	};

	struct directional_light {
		cgv::vec3 direction;
		cgv::vec3 color;
	};

	view* view_ptr = nullptr;
	render_context rctx;

	shader_program bssrdf_shader;
	shader_program euv_shader;
	shader_program shadow_mapping_shader;
	shader_program textured_surface;
	shader_program blur_shader;
	shader_program hair_buffer_shader;
	shader_program hair_resolve_shader;
	shader_program dom_shader;

	frame_buffer fb;
	frame_buffer all_occluders_shadow_map;
	frame_buffer external_occluders_shadow_map;
	frame_buffer blur_fb;
	frame_buffer hair_fb;
	frame_buffer dom_fb;
	frame_buffer scene_fb;

	texture hair_accum;
	texture hair_reveal;
	texture dom_tau_array;
	GLuint dom_tau_array_gl = 0;
	texture scene_texture;
	texture scene_depth;

	cgv::render::texture all_occluders_depth_map;
	cgv::render::texture external_occluders_depth_map;
	cgv::render::texture irradianceBasis[BLUR_PASSES];
	cgv::render::texture temp;
	cgv::render::texture floor_texture;
	cgv::render::texture jitter_tex;

	scene_object<strand_vertex> fur;
	scene_object<vertex> sphere;
	scene_object<vertex> floor;

	texture view_thickness;
	frame_buffer thickness_fb;
	shader_program thickness_shader;

	bool show_euv = false;

	directional_light light;
	float shadow_map_distance = 2.0f;
	float shadow_map_resolution = 256 * 8;

	float widthTip = 0.0035f;
	float widthRoot = 0.012f;

	float sigma[5] = { 1.0f, 2.0f, 4.0f, 8.0f, 16.0f };

	// Hair shading debug / tuning
	float scatterStrength = 1.0f;

	float basis_w0 = 0.35f;
	float basis_w1 = 0.25f;
	float basis_w2 = 0.18f;
	float basis_w3 = 0.14f;
	float basis_w4 = 0.08f;

	float r_lobe_exp = 8.0f;
	float r_grazing_strength = 1.8f;
	float r_shell_mix = 0.02f;

	float tt_lobe_exp = 1.5f;
	float tt_transmission_min = 0.1f;

	float trt_shift = 0.2f;
	float trt_lobe_exp = 6.0f;
	float trt_strength = 1.2f;
	float trt_interior_exp = 0.75f;

	float sss_mask_exp = 1.8f;
	float sss_strength = 0.18f;

	float shell_strength = 0.15f;
	float shell_exp = 1.5f;

	float rim_strength = 0.12f;
	float rim_exp = 2.0f;

	float alpha_root = 0.40f;
	float alpha_tip = 0.18f;

	float tau_scale = 0.5f;
	float tau_noise_min = 0.7f;
	float tau_noise_max = 1.3f;

	float tangent_noise_scale = 10.0f;
	float tangent_noise_angle = 0.25f;
	float tau_noise_scale = 4.0f;

	int dom_layer_count = DOM_LAYER_COUNT;
	float dom_near = 0.1f;
	float dom_far = 10.0f;

	bool debug_show_r = false;
	bool debug_show_tt = false;
	bool debug_show_trt = false;
	bool debug_show_sss = false;
	bool debug_show_shell = false;
	bool debug_show_rim = false;
	bool debug_show_tau = false;
	bool debug_show_alpha = false;

public:
	bssrdf()
		: all_occluders_depth_map("flt16[D]")
		, external_occluders_depth_map("flt16[D]")
		, cgv::base::node("BSSRDF Demo")
	{
		floor.rotation = cgv::vec3(0.0f, 180.0f, 0.0f);
		floor.position = cgv::vec3(0.0f, 0.0f, 0.0f);
		floor.compute_transformation();

		fur.rotation = cgv::vec3(0.0f, 0.0f, 0.0f);
		fur.position = cgv::vec3(0.0f, 0.5f, 0.0f);
		fur.compute_transformation();

		sphere.position = cgv::vec3(1.0f, 3.0f, 1.0f);
		sphere.compute_transformation();

		for (int i = 0; i < BLUR_PASSES; ++i)
			irradianceBasis[i] = cgv::render::texture{ "flt16[R,G,B]" };

		temp = cgv::render::texture{ "flt16[R,G,B]" };
		hair_accum = cgv::render::texture{ "flt16[R,G,B,A]" };
		hair_reveal = cgv::render::texture{ "flt16[R]" };
		scene_texture = cgv::render::texture{ "flt16[R,G,B]" };
		scene_depth = cgv::render::texture{ "flt16[D]" };
		dom_tau_array = cgv::render::texture{ "flt16[R]" };
		view_thickness = cgv::render::texture{ "flt16[R]" };
	}

	std::string get_type_name(void) const { return "bssrdf"; }
	void clear(cgv::render::context& ctx) {}
	bool self_reflect(cgv::reflect::reflection_handler& rh) { return true; }

	void on_set(void* member_ptr) {
		update_member(member_ptr);
		if (this->is_visible())
			post_redraw();
	}

	bool gui_check_value(cgv::gui::control<int>& ctrl) { return true; }
	void gui_value_changed(cgv::gui::control<int>& ctrl) { post_redraw(); }

	void create_gui(void) {
		add_decorator("BSSRDF", "heading");
		add_member_control(this, "Show E(u,v)", show_euv, "check");

		add_decorator("Light Direction", "heading", "level=3");
		add_member_control(this, "X", light.direction[0], "value_slider", "min=-1;max=1;step=0.01;ticks=true");
		add_member_control(this, "Y", light.direction[1], "value_slider", "min=-1;max=1;step=0.01;ticks=true");
		add_member_control(this, "Z", light.direction[2], "value_slider", "min=-1;max=1;step=0.01;ticks=true");
		
		add_decorator("Fur Geometry", "heading", "level=3");
		add_member_control(this, "Fur Tip Width", widthTip, "value_slider", "min=0.001;max=0.15;step=0.001;ticks=true");
		add_member_control(this, "Fur Root Width", widthRoot, "value_slider", "min=0.001;max=0.15;step=0.001;ticks=true");

		add_decorator("Hair Scattering", "heading", "level=3");
		add_member_control(this, "Scatter Strength", scatterStrength, "value_slider", "min=0.0;max=4.0;step=0.01;ticks=true");
		add_member_control(this, "SSS Strength", sss_strength, "value_slider", "min=0.0;max=1.0;step=0.01;ticks=true");
		add_member_control(this, "SSS Mask Exp", sss_mask_exp, "value_slider", "min=0.1;max=4.0;step=0.01;ticks=true");

		add_decorator("Basis Weights", "heading", "level=3");
		add_member_control(this, "w0", basis_w0, "value_slider", "min=0.0;max=1.0;step=0.01;ticks=true");
		add_member_control(this, "w1", basis_w1, "value_slider", "min=0.0;max=1.0;step=0.01;ticks=true");
		add_member_control(this, "w2", basis_w2, "value_slider", "min=0.0;max=1.0;step=0.01;ticks=true");
		add_member_control(this, "w3", basis_w3, "value_slider", "min=0.0;max=1.0;step=0.01;ticks=true");
		add_member_control(this, "w4", basis_w4, "value_slider", "min=0.0;max=1.0;step=0.01;ticks=true");

		add_decorator("R Lobe", "heading", "level=3");
		add_member_control(this, "R Exp", r_lobe_exp, "value_slider", "min=1.0;max=256.0;step=1.0;ticks=true");
		add_member_control(this, "R Grazing", r_grazing_strength, "value_slider", "min=0.0;max=4.0;step=0.01;ticks=true");
		add_member_control(this, "R Shell Mix", r_shell_mix, "value_slider", "min=0.0;max=1.0;step=0.001;ticks=true");

		add_decorator("TT Lobe", "heading", "level=3");
		add_member_control(this, "TT Exp", tt_lobe_exp, "value_slider", "min=0.1;max=8.0;step=0.01;ticks=true");
		add_member_control(this, "TT Min", tt_transmission_min, "value_slider", "min=0.0;max=1.0;step=0.01;ticks=true");

		add_decorator("TRT Lobe", "heading", "level=3");
		add_member_control(this, "TRT Shift", trt_shift, "value_slider", "min=-1.0;max=1.0;step=0.01;ticks=true");
		add_member_control(this, "TRT Exp", trt_lobe_exp, "value_slider", "min=1.0;max=128.0;step=1.0;ticks=true");
		add_member_control(this, "TRT Strength", trt_strength, "value_slider", "min=0.0;max=2.0;step=0.01;ticks=true");
		add_member_control(this, "TRT Interior Exp", trt_interior_exp, "value_slider", "min=0.1;max=4.0;step=0.01;ticks=true");

		add_decorator("Shell / Rim", "heading", "level=3");
		add_member_control(this, "Shell Strength", shell_strength, "value_slider", "min=0.0;max=1.0;step=0.01;ticks=true");
		add_member_control(this, "Shell Exp", shell_exp, "value_slider", "min=0.1;max=4.0;step=0.01;ticks=true");
		add_member_control(this, "Rim Strength", rim_strength, "value_slider", "min=0.0;max=1.0;step=0.01;ticks=true");
		add_member_control(this, "Rim Exp", rim_exp, "value_slider", "min=0.1;max=8.0;step=0.01;ticks=true");

		add_decorator("Alpha", "heading", "level=3");
		add_member_control(this, "Alpha Root", alpha_root, "value_slider", "min=0.0;max=1.0;step=0.01;ticks=true");
		add_member_control(this, "Alpha Tip", alpha_tip, "value_slider", "min=0.0;max=1.0;step=0.01;ticks=true");

		add_decorator("DOM / Noise", "heading", "level=3");
		add_member_control(this, "Tau Scale", tau_scale, "value_slider", "min=0.0;max=4.0;step=0.01;ticks=true");
		add_member_control(this, "Tau Noise Min", tau_noise_min, "value_slider", "min=0.0;max=2.0;step=0.01;ticks=true");
		add_member_control(this, "Tau Noise Max", tau_noise_max, "value_slider", "min=0.0;max=2.0;step=0.01;ticks=true");
		add_member_control(this, "Tangent Noise Scale", tangent_noise_scale, "value_slider", "min=0.1;max=32.0;step=0.1;ticks=true");
		add_member_control(this, "Tangent Noise Angle", tangent_noise_angle, "value_slider", "min=0.0;max=1.0;step=0.001;ticks=true");
		add_member_control(this, "Tau Noise Scale", tau_noise_scale, "value_slider", "min=0.1;max=32.0;step=0.1;ticks=true");

		add_decorator("Debug Views", "heading", "level=3");
		add_member_control(this, "Show R", debug_show_r, "check");
		add_member_control(this, "Show TT", debug_show_tt, "check");
		add_member_control(this, "Show TRT", debug_show_trt, "check");
		add_member_control(this, "Show SSS", debug_show_sss, "check");
		add_member_control(this, "Show Shell", debug_show_shell, "check");
		add_member_control(this, "Show Rim", debug_show_rim, "check");
		add_member_control(this, "Show Tau", debug_show_tau, "check");
		add_member_control(this, "Show Alpha", debug_show_alpha, "check");
	}

	bool init(cgv::render::context& ctx) {
		bool success = true;
		view_ptr = find_view_as_node();

		if (!bssrdf_shader.build_program(ctx, "bssrdf.glpr")) {
			std::cerr << "could not build the bssrdf shader program\n";
			exit(0);
		}
		if (!euv_shader.build_program(ctx, "euv.glpr")) {
			std::cerr << "could not build the euv shader program\n";
			exit(0);
		}
		if (!shadow_mapping_shader.build_program(ctx, "surface_depth.glpr")) {
			std::cerr << "could not build the depth map shader program\n";
			exit(0);
		}
		if (!textured_surface.build_program(ctx, "surface_textured.glpr")) {
			std::cerr << "could not build the textured surface shader program\n";
			exit(0);
		}
		if (!blur_shader.build_program(ctx, "blur.glpr")) {
			std::cerr << "could not build the blur shader program\n";
			exit(0);
		}
		if (!hair_buffer_shader.build_program(ctx, "hair_buffer.glpr")) {
			std::cerr << "could not build hair buffer shader program\n";
			exit(0);
		}
		if (!hair_resolve_shader.build_program(ctx, "hair_resolve.glpr")) {
			std::cerr << "could not build hair resolve shader program\n";
			exit(0);
		}
		if (!dom_shader.build_program(ctx, "dom.glpr")) {
			std::cerr << "could not build DOM shader program\n";
			exit(0);
		}

		if (!thickness_shader.build_program(ctx, "thickness.glpr")) {
			std::cerr << "could not build thickness shader program\n";
			exit(0);
		}

		init_groomed_patch_geometry(fur.verts);
		init_quad_geometry(floor.verts, 10.f);
		init_sphere_geometry(sphere.verts, 0.05f);
		generate_jitter_texture(ctx);

		cgv::render::type_descriptor strand_vec3type =
			cgv::render::element_descriptor_traits<cgv::vec3>::get_type_descriptor(fur.verts[0].center);
		cgv::render::type_descriptor strand_vec2type =
			cgv::render::element_descriptor_traits<cgv::vec2>::get_type_descriptor(fur.verts[0].rootUV);
		cgv::render::type_descriptor strand_floattype =
			cgv::render::element_descriptor_traits<float>::get_type_descriptor(fur.verts[0].VAlong);

		success = fur.vb.create(ctx, &(fur.verts[0]), fur.verts.size()) && success;
		success = fur.va.create(ctx) && success;
		success = fur.va.set_attribute_array(ctx, 0, strand_vec3type, fur.vb, offsetof(strand_vertex, center), fur.verts.size(), sizeof(strand_vertex)) && success;
		success = fur.va.set_attribute_array(ctx, 1, strand_vec3type, fur.vb, offsetof(strand_vertex, tan), fur.verts.size(), sizeof(strand_vertex)) && success;
		success = fur.va.set_attribute_array(ctx, 2, strand_vec2type, fur.vb, offsetof(strand_vertex, rootUV), fur.verts.size(), sizeof(strand_vertex)) && success;
		success = fur.va.set_attribute_array(ctx, 3, strand_floattype, fur.vb, offsetof(strand_vertex, VAlong), fur.verts.size(), sizeof(strand_vertex)) && success;
		success = fur.va.set_attribute_array(ctx, 4, strand_floattype, fur.vb, offsetof(strand_vertex, side), fur.verts.size(), sizeof(strand_vertex)) && success;

		cgv::render::type_descriptor vec3type =
			cgv::render::element_descriptor_traits<cgv::vec3>::get_type_descriptor(floor.verts[0].pos);
		cgv::render::type_descriptor vec2type =
			cgv::render::element_descriptor_traits<cgv::vec2>::get_type_descriptor(floor.verts[0].tcoord);

		success = floor.vb.create(ctx, &(floor.verts[0]), floor.verts.size()) && success;
		success = floor.va.create(ctx) && success;
		success = floor.va.set_attribute_array(ctx, 0, vec3type, floor.vb, 0, floor.verts.size(), sizeof(vertex)) && success;
		success = floor.va.set_attribute_array(ctx, 1, vec3type, floor.vb, sizeof(cgv::vec3), floor.verts.size(), sizeof(vertex)) && success;
		success = floor.va.set_attribute_array(ctx, 2, vec2type, floor.vb, sizeof(cgv::vec3) * 2, floor.verts.size(), sizeof(vertex)) && success;

		success = sphere.vb.create(ctx, &(sphere.verts[0]), sphere.verts.size()) && success;
		success = sphere.va.create(ctx) && success;
		success = sphere.va.set_attribute_array(ctx, 0, vec3type, sphere.vb, 0, sphere.verts.size(), sizeof(vertex)) && success;
		success = sphere.va.set_attribute_array(ctx, 1, vec3type, sphere.vb, sizeof(cgv::vec3), sphere.verts.size(), sizeof(vertex)) && success;
		success = sphere.va.set_attribute_array(ctx, 2, vec2type, sphere.vb, sizeof(cgv::vec3) * 2, sphere.verts.size(), sizeof(vertex)) && success;

		light.direction = cgv::vec3(1.0f, 0.2f, 0.0f);
		light.color = cgv::vec3(1.0f, 1.0f, 1.0f);

		for (int i = 0; i < BLUR_PASSES; ++i) {
			irradianceBasis[i].create(ctx, TextureType::TT_2D, 512, 512);
			irradianceBasis[i].set_wrap_r(TextureWrap::TW_CLAMP_TO_EDGE);
			irradianceBasis[i].set_wrap_s(TextureWrap::TW_CLAMP_TO_EDGE);
			irradianceBasis[i].set_wrap_t(TextureWrap::TW_CLAMP_TO_EDGE);
			irradianceBasis[i].set_min_filter(TextureFilter::TF_LINEAR);
			irradianceBasis[i].set_mag_filter(TextureFilter::TF_LINEAR);
			irradianceBasis[i].set_border_color(1.0f, 1.0f, 1.0f, 1.0f);
		}

		temp.create(ctx, TextureType::TT_2D, 512, 512);
		temp.set_wrap_r(TextureWrap::TW_CLAMP_TO_BORDER);
		temp.set_wrap_s(TextureWrap::TW_CLAMP_TO_BORDER);
		temp.set_wrap_t(TextureWrap::TW_CLAMP_TO_BORDER);
		temp.set_min_filter(TextureFilter::TF_LINEAR);
		temp.set_mag_filter(TextureFilter::TF_LINEAR);
		temp.set_border_color(1.0f, 1.0f, 1.0f, 1.0f);

		hair_accum.create(ctx, TextureType::TT_2D, ctx.get_width(), ctx.get_height());
		hair_accum.set_wrap_r(TextureWrap::TW_CLAMP_TO_BORDER);
		hair_accum.set_wrap_s(TextureWrap::TW_CLAMP_TO_BORDER);
		hair_accum.set_wrap_t(TextureWrap::TW_CLAMP_TO_BORDER);
		hair_accum.set_min_filter(TextureFilter::TF_LINEAR);
		hair_accum.set_mag_filter(TextureFilter::TF_LINEAR);
		hair_accum.set_border_color(1.0f, 1.0f, 1.0f, 1.0f);

		hair_reveal.create(ctx, TextureType::TT_2D, ctx.get_width(), ctx.get_height());
		hair_reveal.set_wrap_r(TextureWrap::TW_CLAMP_TO_BORDER);
		hair_reveal.set_wrap_s(TextureWrap::TW_CLAMP_TO_BORDER);
		hair_reveal.set_wrap_t(TextureWrap::TW_CLAMP_TO_BORDER);
		hair_reveal.set_min_filter(TextureFilter::TF_LINEAR);
		hair_reveal.set_mag_filter(TextureFilter::TF_LINEAR);
		hair_reveal.set_border_color(1.0f, 1.0f, 1.0f, 1.0f);

		scene_texture.create(ctx, TextureType::TT_2D, ctx.get_width(), ctx.get_height());
		scene_texture.set_wrap_r(TextureWrap::TW_CLAMP_TO_BORDER);
		scene_texture.set_wrap_s(TextureWrap::TW_CLAMP_TO_BORDER);
		scene_texture.set_wrap_t(TextureWrap::TW_CLAMP_TO_BORDER);
		scene_texture.set_min_filter(TextureFilter::TF_LINEAR);
		scene_texture.set_mag_filter(TextureFilter::TF_LINEAR);
		scene_texture.set_border_color(1.0f, 1.0f, 1.0f, 1.0f);

		scene_depth.create(ctx, TextureType::TT_2D, ctx.get_width(), ctx.get_height());
		scene_depth.set_wrap_r(TextureWrap::TW_CLAMP_TO_BORDER);
		scene_depth.set_wrap_s(TextureWrap::TW_CLAMP_TO_BORDER);
		scene_depth.set_wrap_t(TextureWrap::TW_CLAMP_TO_BORDER);
		scene_depth.set_min_filter(TextureFilter::TF_LINEAR);
		scene_depth.set_mag_filter(TextureFilter::TF_LINEAR);
		scene_depth.set_border_color(1.0f, 1.0f, 1.0f, 1.0f);

		all_occluders_depth_map.create(ctx, TextureType::TT_2D, shadow_map_resolution, shadow_map_resolution);
		all_occluders_depth_map.set_wrap_s(TW_CLAMP_TO_BORDER);
		all_occluders_depth_map.set_wrap_t(TW_CLAMP_TO_BORDER);
		all_occluders_depth_map.set_min_filter(TF_LINEAR);
		all_occluders_depth_map.set_mag_filter(TF_LINEAR);
		all_occluders_depth_map.set_border_color(1.0f, 1.0f, 1.0f, 1.0f);

		external_occluders_depth_map.create(ctx, TextureType::TT_2D, shadow_map_resolution, shadow_map_resolution);
		external_occluders_depth_map.set_wrap_s(TW_CLAMP_TO_BORDER);
		external_occluders_depth_map.set_wrap_t(TW_CLAMP_TO_BORDER);
		external_occluders_depth_map.set_min_filter(TF_LINEAR);
		external_occluders_depth_map.set_mag_filter(TF_LINEAR);
		external_occluders_depth_map.set_border_color(1.0f, 1.0f, 1.0f, 1.0f);

		glGenTextures(1, &dom_tau_array_gl);
		glObjectLabel(GL_TEXTURE, dom_tau_array_gl, -1, "DOM Tau Texture Array");
		glBindTexture(GL_TEXTURE_2D_ARRAY, dom_tau_array_gl);

		glTexImage3D(
			GL_TEXTURE_2D_ARRAY,
			0,
			GL_R16F,
			(GLsizei)shadow_map_resolution,
			(GLsizei)shadow_map_resolution,
			dom_layer_count,
			0,
			GL_RED,
			GL_FLOAT,
			nullptr
		);

		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

		float border[4] = { 0, 0, 0, 0 };
		glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, border);

		glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

		GLint depth = 0;
		glBindTexture(GL_TEXTURE_2D_ARRAY, dom_tau_array_gl);
		glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_DEPTH, &depth);
		std::cout << "DOM Layers: " << depth << std::endl;
		glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

		GLuint tex = (GLuint)(uintptr_t)dom_tau_array.handle;

		GLint target = 0;
		glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
		glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_DEPTH, &target);

		std::cout << "DOM layers: " << target << std::endl;

		view_thickness.create(ctx, TextureType::TT_2D, ctx.get_width(), ctx.get_height());
		view_thickness.set_wrap_r(TextureWrap::TW_CLAMP_TO_BORDER);
		view_thickness.set_wrap_s(TextureWrap::TW_CLAMP_TO_BORDER);
		view_thickness.set_wrap_t(TextureWrap::TW_CLAMP_TO_BORDER);
		view_thickness.set_min_filter(TextureFilter::TF_LINEAR);
		view_thickness.set_mag_filter(TextureFilter::TF_LINEAR);
		view_thickness.set_border_color(0.0f, 0.0f, 0.0f, 0.0f);

		success &= thickness_fb.create(ctx, ctx.get_width(), ctx.get_height());
		success &= thickness_fb.attach(ctx, view_thickness, 0, 0);

		success &= fb.create(ctx, 512, 512);
		success &= fb.attach(ctx, irradianceBasis[0], 0, 0);

		success &= scene_fb.create(ctx, ctx.get_width(), ctx.get_height());
		success &= scene_fb.attach(ctx, scene_texture, 0, 0);
		success &= scene_fb.attach(ctx, scene_depth, 0);

		success &= hair_fb.create(ctx, ctx.get_width(), ctx.get_height());
		success &= hair_fb.attach(ctx, hair_accum, 0, 0);
		success &= hair_fb.attach(ctx, hair_reveal, 0, 1);
		success &= hair_fb.attach(ctx, scene_depth, 0);

		success &= all_occluders_shadow_map.create(ctx, shadow_map_resolution, shadow_map_resolution);
		success &= all_occluders_shadow_map.attach(ctx, all_occluders_depth_map);

		success &= external_occluders_shadow_map.create(ctx, shadow_map_resolution, shadow_map_resolution);
		success &= external_occluders_shadow_map.attach(ctx, external_occluders_depth_map);

		success &= dom_fb.create(ctx, shadow_map_resolution, shadow_map_resolution);
		GLuint dom_fbo = static_cast<GLuint>(reinterpret_cast<uintptr_t>(dom_fb.handle));
		glObjectLabel(GL_FRAMEBUFFER, dom_fbo, -1, "DOM Layered FBO");

		success &= blur_fb.create(ctx, 512, 512);

		read_texture(ctx, floor.albedo_tex, "C:/dev/BSSRDF/bssrdf/res/floor_albedo.png");
		read_texture(ctx, sphere.albedo_tex, "C:/dev/BSSRDF/bssrdf/res/floor_albedo.png");
		read_texture(ctx, fur.albedo_tex, "C:/dev/BSSRDF/bssrdf/res/fur_color.png");

		glObjectLabel(GL_TEXTURE, dom_tau_array_gl, -1, "DOM Tau Texture Array");

		glObjectLabel(GL_FRAMEBUFFER, (GLuint)(uintptr_t)dom_fb.handle - 1, -1, "DOM FBO");
		glObjectLabel(GL_FRAMEBUFFER, (GLuint)(uintptr_t)scene_fb.handle - 1, -1, "Scene FBO");
		glObjectLabel(GL_FRAMEBUFFER, (GLuint)(uintptr_t)hair_fb.handle - 1, -1, "Hair Buffer FBO");
		glObjectLabel(GL_FRAMEBUFFER, (GLuint)(uintptr_t)thickness_fb.handle - 1, -1, "View Thickness FBO");
		glObjectLabel(GL_FRAMEBUFFER, (GLuint)(uintptr_t)all_occluders_shadow_map.handle - 1, -1, "Shadow FBO - All Occluders");
		glObjectLabel(GL_FRAMEBUFFER, (GLuint)(uintptr_t)external_occluders_shadow_map.handle - 1, -1, "Shadow FBO - External Occluders");
		glObjectLabel(GL_FRAMEBUFFER, (GLuint)(uintptr_t)fb.handle - 1, -1, "Irradiance FBO");
		glObjectLabel(GL_FRAMEBUFFER, (GLuint)(uintptr_t)blur_fb.handle - 1, -1, "Blur FBO");

		rctx.ctx = &ctx;
		return success;
	}

	bool read_texture(context& ctx, texture& tex, const std::string& file_name) {
		cgv::data::data_format df;
		cgv::data::data_view dv;
		bool success = tex.create_from_image(df, dv, ctx, file_name);
		tex.set_wrap_s(TW_REPEAT);
		tex.set_wrap_t(TW_REPEAT);
		tex.set_min_filter(TF_ANISOTROP, 16.0f);
		tex.set_mag_filter(TF_LINEAR);
		tex.generate_mipmaps(ctx);
		return success;
	}

	void init_frame(cgv::render::context& ctx) {}

	void draw(cgv::render::context& ctx) {
		if (!view_ptr)
			return;

		all_occluders_depth_map.set_compare_mode(true);
		external_occluders_depth_map.set_compare_mode(true);

		cgv::vec3 light_direction = normalize(light.direction);
		cgv::mat4 light_projection = cgv::math::ortho4(-3.0f, 3.0f, -3.0f, 3.0f, 0.1f, 10.0f);
		cgv::mat4 light_view = cgv::math::look_at4(5.0f * light_direction, cgv::vec3(1.0f, 1.0f, 1.0f), cgv::vec3(0.0f, 1.0f, 0.0f));
		cgv::mat4 light_matrix = light_projection * light_view;

		rctx.store_view();
		rctx.light_matrix = light_matrix;

		// SHADOW MAP: all occluders
		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 10, -1, "Shadow Map - All Occluders");
		ctx.push_window_transformation_array();
		ctx.set_viewport(cgv::ivec4(0, 0, shadow_map_resolution, shadow_map_resolution));

		all_occluders_shadow_map.enable(ctx);
		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
		glDisable(GL_CULL_FACE);
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glDepthMask(GL_TRUE);
		glDisable(GL_STENCIL_TEST);
		glClearDepth(1.0);
		glClear(GL_DEPTH_BUFFER_BIT);

		shadow_mapping_shader.enable(ctx);
		shadow_mapping_shader.set_uniform(ctx, "u_widthTip", widthTip);
		shadow_mapping_shader.set_uniform(ctx, "u_widthRoot", widthRoot);
		shadow_mapping_shader.set_uniform(ctx, "u_lightDirWS", light.direction);

		floor.draw_depth(rctx, shadow_mapping_shader);
		fur.draw_depth(rctx, shadow_mapping_shader);
		sphere.draw_depth(rctx, shadow_mapping_shader);

		shadow_mapping_shader.disable(ctx);
		all_occluders_shadow_map.disable(ctx);
		glPopDebugGroup();

		// SHADOW MAP: external occluders
		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 20, -1, "Shadow Map - External Occluders");
		external_occluders_shadow_map.enable(ctx);
		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
		glDisable(GL_CULL_FACE);
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glDepthMask(GL_TRUE);
		glDisable(GL_STENCIL_TEST);
		glClearDepth(1.0);
		glClear(GL_DEPTH_BUFFER_BIT);

		shadow_mapping_shader.enable(ctx);
		shadow_mapping_shader.set_uniform(ctx, "u_lightDirWS", light.direction);
		shadow_mapping_shader.set_uniform(ctx, "u_widthTip", widthTip);
		shadow_mapping_shader.set_uniform(ctx, "u_widthRoot", widthRoot);

		floor.draw_depth(rctx, shadow_mapping_shader);
		sphere.draw_depth(rctx, shadow_mapping_shader);

		shadow_mapping_shader.disable(ctx);
		external_occluders_shadow_map.disable(ctx);

		glEnable(GL_SCISSOR_TEST);
		glEnable(GL_CULL_FACE);
		ctx.pop_window_transformation_array();

		ctx.push_modelview_matrix();
		glPopDebugGroup();


		// DOM PASS
		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "DOM Pass");
		dom_shader.enable(ctx);

		ctx.push_window_transformation_array();
		ctx.set_viewport(cgv::ivec4(0, 0, shadow_map_resolution, shadow_map_resolution));

		glDisable(GL_CULL_FACE);
		glDisable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);
		glEnable(GL_BLEND);
		glBlendEquation(GL_FUNC_ADD);
		glBlendFunc(GL_ONE, GL_ONE);

		// Clear all layers of the DOM texture array
		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 1, -1, "DOM Clear");
		clear_dom_texture_array(ctx);
		glPopDebugGroup();

		// Bind DOM FBO and attach the WHOLE array texture as layered target
		GLuint fbo = static_cast<GLuint>(reinterpret_cast<uintptr_t>(dom_fb.handle));
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, dom_tau_array_gl, 0);

		GLenum buf = GL_COLOR_ATTACHMENT0;
		glDrawBuffers(1, &buf);

		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			std::cout << "DOM FBO ERROR: " << status << std::endl;
		}

		dom_shader.set_uniform(ctx, "u_lightSpaceMatrix", rctx.light_matrix * fur.transformation_matrix);
		dom_shader.set_uniform(ctx, "u_model_matrix", fur.transformation_matrix);
		dom_shader.set_uniform(ctx, "u_lightDirWS", light.direction);
		dom_shader.set_uniform(ctx, "widthTip", widthTip);
		dom_shader.set_uniform(ctx, "widthRoot", widthRoot);
		dom_shader.set_uniform(ctx, "u_domLayerCount", dom_layer_count);
		dom_shader.set_uniform(ctx, "u_domNear", dom_near);
		dom_shader.set_uniform(ctx, "u_domFar", dom_far);

		fur.set_modelview(rctx);
		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 2, -1, "DOM Draw Fur");
		fur.va.enable(ctx);
		glDrawArrays(GL_TRIANGLES, 0, (GLsizei)fur.verts.size());
		fur.va.disable(ctx);
		glPopDebugGroup();

		// Unbind framebuffer
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		dom_shader.disable(ctx);
		glPopDebugGroup();

		glDisable(GL_BLEND);
		glDepthMask(GL_TRUE);

		ctx.pop_window_transformation_array();

		// E(u,v)
		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 40, -1, "Irradiance E(u,v) Pass");
		euv_shader.enable(ctx);
		euv_shader.set_uniform(ctx, "u_lightDirWS", light.direction);
		euv_shader.set_uniform(ctx, "u_lightRadiance", light.color);
		euv_shader.set_uniform(ctx, "u_boundaryNormalWS", cgv::vec3(0.0f, 1.0f, 0.0f));
		euv_shader.set_uniform(ctx, "u_ambientIrradiance", cgv::vec3(0.1f));
		euv_shader.set_uniform(ctx, "u_lightSpaceMatrix", rctx.light_matrix);
		euv_shader.set_uniform(ctx, "u_widthTip", widthTip);
		euv_shader.set_uniform(ctx, "u_widthRoot", widthRoot);
		euv_shader.set_uniform(ctx, "shadowMap", 0);

		external_occluders_depth_map.enable(ctx, 0);
		fb.enable(ctx, 0);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		fb.disable(ctx);
		external_occluders_depth_map.disable(ctx);
		euv_shader.disable(ctx);
		glPopDebugGroup();

		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 50, -1, "Irradiance Blur Passes");
		for (int i = 1; i < BLUR_PASSES; ++i) {
			blur(ctx, irradianceBasis[0], cgv::vec2(1.0f, 0.0f), sigma[i], temp);
			blur(ctx, temp, cgv::vec2(0.0f, 1.0f), sigma[i], irradianceBasis[i]);
		}
		glPopDebugGroup();

		// OPAQUE SCENE
		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 60, -1, "Opaque Scene Pass");
		scene_fb.enable(ctx, 0);
		glClearColor(1.0, 1.0, 1.0, 1.0);
		glClearDepth(1.0);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

		textured_surface.enable(ctx);
		textured_surface.set_uniform(ctx, "shadowMap", 0);
		textured_surface.set_uniform(ctx, "jitter_tex", 1);
		textured_surface.set_uniform(ctx, "uTex", 2);
		textured_surface.set_uniform(ctx, "u_lightdir", light.direction);
		textured_surface.set_uniform(ctx, "u_lightColor", light.color);
		textured_surface.set_uniform(ctx, "u_ambient", cgv::vec3(0.2f, 0.2f, 0.2f));
		textured_surface.set_uniform(ctx, "u_lightSpaceMatrix", rctx.light_matrix);
		textured_surface.set_uniform(ctx, "u_widthTip", widthTip);
		textured_surface.set_uniform(ctx, "u_widthRoot", widthRoot);

		all_occluders_depth_map.enable(ctx, 0);
		jitter_tex.enable(ctx, 1);

		floor.albedo_tex.enable(ctx, 2);
		floor.draw_bound(rctx, textured_surface);
		floor.albedo_tex.disable(ctx);

		sphere.albedo_tex.enable(ctx, 2);
		sphere.draw_bound(rctx, textured_surface);
		sphere.albedo_tex.disable(ctx);

		all_occluders_depth_map.disable(ctx);
		jitter_tex.disable(ctx);

		textured_surface.disable(ctx);
		scene_fb.disable(ctx);
		ctx.pop_modelview_matrix();
		glPopDebugGroup();

		// VIEW-THICKNESS PASS
		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 70, -1, "View Thickness Pass");
		thickness_fb.enable(ctx, 0);
		glViewport(0, 0, ctx.get_width(), ctx.get_height());

		const float zero1[4] = { 0, 0, 0, 0 };
		glClearBufferfv(GL_COLOR, 0, zero1);

		glDisable(GL_CULL_FACE);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glDepthMask(GL_FALSE);
		glEnable(GL_BLEND);
		glBlendEquation(GL_FUNC_ADD);
		glBlendFunc(GL_ONE, GL_ONE);

		thickness_shader.enable(ctx);

		thickness_shader.set_uniform(ctx, "u_model_matrix", fur.transformation_matrix);
		thickness_shader.set_uniform(ctx, "u_model_normal_matrix", rctx.get_normal_matrix(fur.transformation_matrix));
		thickness_shader.set_uniform(ctx, "u_widthTip", widthTip);
		thickness_shader.set_uniform(ctx, "u_widthRoot", widthRoot);

		fur.set_modelview(rctx);
		fur.va.enable(ctx);
		glDrawArrays(GL_TRIANGLES, 0, (GLsizei)fur.verts.size());
		fur.va.disable(ctx);

		thickness_shader.disable(ctx);
		thickness_fb.disable(ctx);
		glPopDebugGroup();

		glDisable(GL_BLEND);
		glDepthMask(GL_TRUE);

		// HAIR BUFFER PASS
		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 80, -1, "Hair Buffer Pass");
		glDisable(GL_BLEND);

		hair_buffer_shader.enable(ctx);
		hair_fb.enable(ctx, 0, 1);
		glViewport(0, 0, ctx.get_width(), ctx.get_height());

		const float zero4[4] = { 0, 0, 0, 0 };
		glClearBufferfv(GL_COLOR, 0, zero4);
		glClearBufferfv(GL_COLOR, 1, zero4);

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glDepthMask(GL_FALSE);
		glDisable(GL_CULL_FACE);
		glEnable(GL_BLEND);

		glBlendEquationi(0, GL_FUNC_ADD);
		glBlendFunci(0, GL_ONE, GL_ONE);
		glBlendEquationi(1, GL_FUNC_ADD);
		glBlendFunci(1, GL_ONE, GL_ONE);

		glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glColorMaski(1, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

		irradianceBasis[0].enable(ctx, 0);
		irradianceBasis[1].enable(ctx, 1);
		irradianceBasis[2].enable(ctx, 2);
		irradianceBasis[3].enable(ctx, 3);
		irradianceBasis[4].enable(ctx, 4);
		fur.albedo_tex.enable(ctx, 5);
		glActiveTexture(GL_TEXTURE6);
		glBindTexture(GL_TEXTURE_2D_ARRAY, dom_tau_array_gl);
		glActiveTexture(GL_TEXTURE0);
		view_thickness.enable(ctx, 7);

		hair_buffer_shader.set_uniform(ctx, "u_IrradianceBasis0", 0);
		hair_buffer_shader.set_uniform(ctx, "u_IrradianceBasis1", 1);
		hair_buffer_shader.set_uniform(ctx, "u_IrradianceBasis2", 2);
		hair_buffer_shader.set_uniform(ctx, "u_IrradianceBasis3", 3);
		hair_buffer_shader.set_uniform(ctx, "u_IrradianceBasis4", 4);
		hair_buffer_shader.set_uniform(ctx, "fur_color", 5);
		hair_buffer_shader.set_uniform(ctx, "u_domTau", 6);
		hair_buffer_shader.set_uniform(ctx, "u_viewThickness", 7);
		hair_buffer_shader.set_uniform(ctx, "u_domLayerCount", dom_layer_count);
		hair_buffer_shader.set_uniform(ctx, "u_domNear", dom_near);
		hair_buffer_shader.set_uniform(ctx, "u_domFar", dom_far);

		hair_buffer_shader.set_uniform(ctx, "u_model_matrix", fur.transformation_matrix);
		hair_buffer_shader.set_uniform(ctx, "u_model_normal_matrix", rctx.get_normal_matrix(fur.transformation_matrix));
		hair_buffer_shader.set_uniform(ctx, "u_lightSpaceMatrix", rctx.light_matrix * fur.transformation_matrix);

		hair_buffer_shader.set_uniform(ctx, "u_scatterStrength", 1.0f);
		hair_buffer_shader.set_uniform(ctx, "u_w0", 0.35f);
		hair_buffer_shader.set_uniform(ctx, "u_w1", 0.25f);
		hair_buffer_shader.set_uniform(ctx, "u_w2", 0.18f);
		hair_buffer_shader.set_uniform(ctx, "u_w3", 0.14f);
		hair_buffer_shader.set_uniform(ctx, "u_w4", 0.08f);

		hair_buffer_shader.set_uniform(ctx, "u_widthTip", widthTip);
		hair_buffer_shader.set_uniform(ctx, "u_widthRoot", widthRoot);
		hair_buffer_shader.set_uniform(ctx, "u_lightdir", light.direction);
		hair_buffer_shader.set_uniform(ctx, "u_lightColor", light.color);
		hair_buffer_shader.set_uniform(ctx, "u_ambient", cgv::vec3(0.2f, 0.2f, 0.2f));

		hair_buffer_shader.set_uniform(ctx, "u_scatterStrength", scatterStrength);

		hair_buffer_shader.set_uniform(ctx, "u_w0", basis_w0);
		hair_buffer_shader.set_uniform(ctx, "u_w1", basis_w1);
		hair_buffer_shader.set_uniform(ctx, "u_w2", basis_w2);
		hair_buffer_shader.set_uniform(ctx, "u_w3", basis_w3);
		hair_buffer_shader.set_uniform(ctx, "u_w4", basis_w4);

		hair_buffer_shader.set_uniform(ctx, "u_rLobeExp", r_lobe_exp);
		hair_buffer_shader.set_uniform(ctx, "u_rGrazingStrength", r_grazing_strength);
		hair_buffer_shader.set_uniform(ctx, "u_rShellMix", r_shell_mix);

		hair_buffer_shader.set_uniform(ctx, "u_ttLobeExp", tt_lobe_exp);
		hair_buffer_shader.set_uniform(ctx, "u_ttTransmissionMin", tt_transmission_min);

		hair_buffer_shader.set_uniform(ctx, "u_trtShift", trt_shift);
		hair_buffer_shader.set_uniform(ctx, "u_trtLobeExp", trt_lobe_exp);
		hair_buffer_shader.set_uniform(ctx, "u_trtStrength", trt_strength);
		hair_buffer_shader.set_uniform(ctx, "u_trtInteriorExp", trt_interior_exp);

		hair_buffer_shader.set_uniform(ctx, "u_sssMaskExp", sss_mask_exp);
		hair_buffer_shader.set_uniform(ctx, "u_sssStrength", sss_strength);

		hair_buffer_shader.set_uniform(ctx, "u_shellStrength", shell_strength);
		hair_buffer_shader.set_uniform(ctx, "u_shellExp", shell_exp);
		hair_buffer_shader.set_uniform(ctx, "u_rimStrength", rim_strength);
		hair_buffer_shader.set_uniform(ctx, "u_rimExp", rim_exp);

		hair_buffer_shader.set_uniform(ctx, "u_alphaRoot", alpha_root);
		hair_buffer_shader.set_uniform(ctx, "u_alphaTip", alpha_tip);

		hair_buffer_shader.set_uniform(ctx, "u_tauScale", tau_scale);
		hair_buffer_shader.set_uniform(ctx, "u_tauNoiseMin", tau_noise_min);
		hair_buffer_shader.set_uniform(ctx, "u_tauNoiseMax", tau_noise_max);
		hair_buffer_shader.set_uniform(ctx, "u_tangentNoiseScale", tangent_noise_scale);
		hair_buffer_shader.set_uniform(ctx, "u_tangentNoiseAngle", tangent_noise_angle);
		hair_buffer_shader.set_uniform(ctx, "u_tauNoiseScale", tau_noise_scale);

		hair_buffer_shader.set_uniform(ctx, "u_debugShowR", debug_show_r);
		hair_buffer_shader.set_uniform(ctx, "u_debugShowTT", debug_show_tt);
		hair_buffer_shader.set_uniform(ctx, "u_debugShowTRT", debug_show_trt);
		hair_buffer_shader.set_uniform(ctx, "u_debugShowSSS", debug_show_sss);
		hair_buffer_shader.set_uniform(ctx, "u_debugShowShell", debug_show_shell);
		hair_buffer_shader.set_uniform(ctx, "u_debugShowRim", debug_show_rim);
		hair_buffer_shader.set_uniform(ctx, "u_debugShowTau", debug_show_tau);
		hair_buffer_shader.set_uniform(ctx, "u_debugShowAlpha", debug_show_alpha);

		fur.set_modelview(rctx);
		fur.va.enable(ctx);
		glDrawArrays(GL_TRIANGLES, 0, (GLsizei)fur.verts.size());
		fur.va.disable(ctx);

		glActiveTexture(GL_TEXTURE6);
		glBindTexture(GL_TEXTURE_2D_ARRAY, dom_tau_array_gl);
		glActiveTexture(GL_TEXTURE0);
		fur.albedo_tex.disable(ctx);
		irradianceBasis[4].disable(ctx);
		irradianceBasis[3].disable(ctx);
		irradianceBasis[2].disable(ctx);
		irradianceBasis[1].disable(ctx);
		irradianceBasis[0].disable(ctx);

		hair_buffer_shader.disable(ctx);
		hair_fb.disable(ctx);

		glDepthMask(GL_TRUE);
		glDisable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glPopDebugGroup();

		// HAIR RESOLVE
		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 90, -1, "Hair Resolve Pass");
		hair_resolve_shader.enable(ctx);

		hair_accum.enable(ctx, 0);
		hair_reveal.enable(ctx, 1);
		scene_texture.enable(ctx, 2);

		hair_resolve_shader.set_uniform(ctx, "u_AccumTex", 0);
		hair_resolve_shader.set_uniform(ctx, "u_logRTex", 1);
		hair_resolve_shader.set_uniform(ctx, "u_sceneColor", 2);
		hair_resolve_shader.set_uniform(ctx, "u_widthTip", widthTip);
		hair_resolve_shader.set_uniform(ctx, "u_widthRoot", widthRoot);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		scene_texture.disable(ctx);
		hair_reveal.disable(ctx);
		hair_accum.disable(ctx);

		hair_resolve_shader.disable(ctx);
		glPopDebugGroup();
	}

	void blur(cgv::render::context& ctx, cgv::render::texture& src, cgv::vec2 direction, float sigma, cgv::render::texture& dest)
	{
		blur_fb.attach(ctx, dest, 0, 0);
		blur_fb.enable(ctx);

		ctx.push_window_transformation_array();
		ctx.set_viewport(cgv::ivec4(0, 0, 512, 512));

		glDisable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
		glDisable(GL_SCISSOR_TEST);
		glDepthMask(GL_FALSE);

		blur_shader.enable(ctx);
		src.enable(ctx, 0);
		blur_shader.set_uniform(ctx, "irrad", 0);
		blur_shader.set_uniform(ctx, "u_dir", direction);
		blur_shader.set_uniform(ctx, "u_sigma", sigma);
		blur_shader.set_uniform(ctx, "u_texelSize", cgv::vec2(1.0f / 512.0f, 1.0f / 512.0f));

		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		src.disable(ctx);
		blur_shader.disable(ctx);
		blur_fb.disable(ctx);

		glEnable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glEnable(GL_SCISSOR_TEST);
		glDepthMask(GL_TRUE);

		ctx.pop_window_transformation_array();
	}

	bool generate_jitter_texture(cgv::render::context& ctx) {
		bool success = true;

		std::mt19937 rng(42);
		std::uniform_real_distribution<float> distr(-1.0f, 1.0f);

		int size = 16;
		int samples_u = 8;
		int samples_v = 8;

		std::vector<cgv::vec4> jitter_data(size * size * samples_u * samples_v / 2);

		for (int i = 0; i < size; ++i) {
			for (int j = 0; j < size; ++j) {
				for (int k = 0; k < samples_u * samples_v / 2; ++k) {
					int x = k % (samples_u / 2);
					int y = (samples_v - 1) - k / (samples_u / 2);

					cgv::vec4 v;
					v[0] = (float)(x * 2 + 0.5f) / samples_u;
					v[1] = (float)(y + 0.5f) / samples_v;
					v[2] = (float)(x * 2 + 1 + 0.5f) / samples_u;
					v[3] = v[1];

					v[0] += distr(rng) * (0.5f / samples_u);
					v[1] += distr(rng) * (0.5f / samples_v);
					v[2] += distr(rng) * (0.5f / samples_u);
					v[3] += distr(rng) * (0.5f / samples_v);

					jitter_data[(k * size * size + j * size + i)] = static_cast<cgv::vec4>(127.0f * v);
				}
			}
		}

		if (jitter_tex.is_created())
			jitter_tex.destruct(ctx);

		cgv::data::data_view jitter_dv =
			cgv::data::data_view(
				new cgv::data::data_format(size, size, samples_u * samples_v / 2, cgv::type::info::TI_INT8, cgv::data::CF_RGBA),
				jitter_data.data());

		success &= jitter_tex.create(ctx, jitter_dv, 0);
		jitter_tex.set_wrap_s(TW_REPEAT);
		jitter_tex.set_wrap_t(TW_REPEAT);
		jitter_tex.set_wrap_r(TW_REPEAT);
		jitter_tex.set_min_filter(TF_NEAREST);
		jitter_tex.set_mag_filter(TF_NEAREST);

		return success;
	}

	static inline cgv::vec3 safe_normalize(const cgv::vec3& v) {
		float len = cgv::math::length(v);
		if (len < 1e-8f) return cgv::vec3(1.0f, 0.0f, 0.0f);
		return v / len;
	}

	static inline int resolve_obj_index_1based(int idx, int count) {
		if (idx > 0) return idx - 1;
		if (idx < 0) return count + idx;
		return -1;
	}

	struct Segment { int a, b; };

	void init_strands_geometry_from_segment_obj(const char* objPath, std::vector<strand_vertex>& verts)
	{
		const uint32_t STRAND_POINTS = 3;
		const cgv::vec3 POSITION_OFFSET = cgv::vec3(1.0f, 0.0f, 0.0f);

		std::ifstream in(objPath);
		if (!in.is_open())
			return;

		std::vector<cgv::vec3> positions;
		std::vector<std::pair<int, int>> rawSegments;

		positions.reserve(50000);
		rawSegments.reserve(40000);

		std::string line;
		while (std::getline(in, line)) {
			if (line.empty() || line[0] == '#')
				continue;

			std::istringstream ss(line);
			std::string tag;
			ss >> tag;

			if (tag == "v") {
				float x, y, z;
				ss >> x >> y >> z;
				positions.push_back(cgv::vec3(x, y, z));
			}
			else if (tag == "l") {
				int ia = 0, ib = 0;
				ss >> ia >> ib;
				if (!ss.fail())
					rawSegments.emplace_back(ia, ib);
			}
		}

		if (positions.empty() || rawSegments.empty())
			return;

		int minPositiveRef = std::numeric_limits<int>::max();
		int maxPositiveRef = std::numeric_limits<int>::min();

		for (const auto& s : rawSegments) {
			if (s.first > 0) {
				minPositiveRef = std::min(minPositiveRef, s.first);
				maxPositiveRef = std::max(maxPositiveRef, s.first);
			}
			if (s.second > 0) {
				minPositiveRef = std::min(minPositiveRef, s.second);
				maxPositiveRef = std::max(maxPositiveRef, s.second);
			}
		}

		int globalBias = 0;
		if (minPositiveRef != std::numeric_limits<int>::max()) {
			const int referencedSpan = maxPositiveRef - minPositiveRef + 1;
			if (referencedSpan == (int)positions.size() && minPositiveRef != 1)
				globalBias = minPositiveRef - 1;
		}

		auto resolve_index = [&](int idx1Based) -> int {
			if (idx1Based < 0)
				return (int)positions.size() + idx1Based;
			return idx1Based - 1 - globalBias;
			};

		std::vector<Segment> segments;
		segments.reserve(rawSegments.size());

		for (const auto& rs : rawSegments) {
			int a = resolve_index(rs.first);
			int b = resolve_index(rs.second);
			if (a < 0 || b < 0 || a >= (int)positions.size() || b >= (int)positions.size())
				continue;
			segments.push_back({ a, b });
		}

		if (segments.empty())
			return;

		std::vector<std::vector<int>> adj(positions.size());
		for (const Segment& s : segments) {
			adj[s.a].push_back(s.b);
			adj[s.b].push_back(s.a);
		}

		std::vector<uint8_t> visited(positions.size(), 0);
		std::vector<std::vector<int>> strands;
		strands.reserve(20000);

		for (int v = 0; v < (int)positions.size(); ++v) {
			if ((int)adj[v].size() != 1 || visited[v])
				continue;

			std::vector<int> path;
			path.reserve(8);

			int prev = -1;
			int cur = v;

			while (true) {
				visited[cur] = 1;
				path.push_back(cur);

				int next = -1;
				for (int nb : adj[cur]) {
					if (nb != prev) {
						next = nb;
						break;
					}
				}

				if (next < 0)
					break;

				prev = cur;
				cur = next;

				if ((int)adj[cur].size() == 1) {
					visited[cur] = 1;
					path.push_back(cur);
					break;
				}
			}

			if (path.size() >= 2)
				strands.push_back(std::move(path));
		}

		if (strands.empty())
			return;

		cgv::vec3 bbMin(1e30f, 1e30f, 1e30f);
		cgv::vec3 bbMax(-1e30f, -1e30f, -1e30f);

		auto endpoint_root_index = [&](const std::vector<int>& path) -> int {
			const cgv::vec3& p0 = positions[path.front()];
			const cgv::vec3& p1 = positions[path.back()];
			return (p0.y() <= p1.y()) ? path.front() : path.back();
			};

		for (const auto& path : strands) {
			int r = endpoint_root_index(path);
			const cgv::vec3& rp = positions[r];
			bbMin = cgv::vec3(std::min(bbMin.x(), rp.x()), std::min(bbMin.y(), rp.y()), std::min(bbMin.z(), rp.z()));
			bbMax = cgv::vec3(std::max(bbMax.x(), rp.x()), std::max(bbMax.y(), rp.y()), std::max(bbMax.z(), rp.z()));
		}

		cgv::vec3 bbSpan = bbMax - bbMin;
		if (std::abs(bbSpan.x()) < 1e-8f) bbSpan.x() = 1.0f;
		if (std::abs(bbSpan.z()) < 1e-8f) bbSpan.z() = 1.0f;

		verts.clear();
		verts.reserve(strands.size() * STRAND_POINTS * 2);

		auto sample_polyline_arc = [&](const std::vector<int>& path, float u01) -> cgv::vec3 {
			if (path.size() == 1)
				return positions[path[0]];

			float total = 0.0f;
			std::vector<float> cum(path.size(), 0.0f);

			for (size_t i = 1; i < path.size(); ++i) {
				total += cgv::math::length(positions[path[i]] - positions[path[i - 1]]);
				cum[i] = total;
			}

			if (total < 1e-8f)
				return positions[path.front()];

			float target = u01 * total;
			auto it = std::lower_bound(cum.begin(), cum.end(), target);
			size_t i = (size_t)std::clamp<int>((int)(it - cum.begin()), 1, (int)path.size() - 1);

			float segStart = cum[i - 1];
			float segEnd = cum[i];
			float t = (segEnd > segStart) ? (target - segStart) / (segEnd - segStart) : 0.0f;

			const cgv::vec3& a = positions[path[i - 1]];
			const cgv::vec3& b = positions[path[i]];
			return a + (b - a) * t;
			};

		for (auto path : strands) {
			const cgv::vec3& p0 = positions[path.front()];
			const cgv::vec3& p1 = positions[path.back()];

			if (p0.y() > p1.y())
				std::reverse(path.begin(), path.end());

			cgv::vec3 rootPos3 = positions[path.front()];
			cgv::vec2 rootUV(
				(rootPos3.x() - bbMin.x()) / bbSpan.x(),
				(rootPos3.z() - bbMin.z()) / bbSpan.z()
			);

			cgv::vec3 samples[STRAND_POINTS];
			for (uint32_t i = 0; i < STRAND_POINTS; ++i) {
				float u = (STRAND_POINTS == 1) ? 0.0f : float(i) / float(STRAND_POINTS - 1);
				samples[i] = sample_polyline_arc(path, u);
			}

			for (uint32_t i = 0; i < STRAND_POINTS; ++i) {
				float V = (STRAND_POINTS == 1) ? 0.0f : float(i) / float(STRAND_POINTS - 1);

				cgv::vec3 tan;
				if (i == 0)
					tan = safe_normalize(samples[1] - samples[0]);
				else if (i + 1 == STRAND_POINTS)
					tan = safe_normalize(samples[i] - samples[i - 1]);
				else
					tan = safe_normalize(samples[i + 1] - samples[i - 1]);

				cgv::vec3 center = samples[i] - POSITION_OFFSET;

				strand_vertex vL{};
				vL.center = center;
				vL.tan = tan;
				vL.rootUV = rootUV;
				vL.VAlong = V;
				vL.side = -1.0f;

				strand_vertex vR = vL;
				vR.side = 1.0f;

				verts.push_back(vL);
				verts.push_back(vR);
			}
		}
	}

	void init_groomed_patch_geometry(std::vector<strand_vertex>& verts) {
		const uint32_t STRAND_AMOUNT = 50000;
		const uint32_t STRAND_POINTS = 12;
		const float PATCH_WIDTH = 1.6f;
		const float PATCH_DEPTH = 1.6f;
		const float LENGTH_MIN = 0.22f;
		const float LENGTH_MAX = 0.90f;

		const cgv::vec3 BASE_DIR =
			cgv::math::normalize(cgv::vec3(1.0f, 0.22f, 0.12f));

		verts.clear();
		verts.reserve(STRAND_AMOUNT * (STRAND_POINTS - 1) * 6);

		std::mt19937 rng(1337);
		std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
		std::uniform_real_distribution<float> jitter(-1.0f, 1.0f);

		auto rotate_y = [](const cgv::vec3& v, float a) {
			float c = std::cos(a);
			float s = std::sin(a);
			return cgv::vec3(c * v.x() + s * v.z(), v.y(), -s * v.x() + c * v.z());
			};

		auto rotate_z = [](const cgv::vec3& v, float a) {
			float c = std::cos(a);
			float s = std::sin(a);
			return cgv::vec3(c * v.x() - s * v.y(), s * v.x() + c * v.y(), v.z());
			};

		auto smoothstep01 = [](float x) {
			x = std::clamp(x, 0.0f, 1.0f);
			return x * x * (3.0f - 2.0f * x);
			};

		auto hash21 = [](const cgv::vec2& p) {
			float n = std::sin(p.x() * 127.1f + p.y() * 311.7f) * 43758.5453f;
			return n - std::floor(n);
			};

		auto hash22 = [&](const cgv::vec2& p) {
			return cgv::vec2(
				hash21(p + cgv::vec2(17.0f, 3.0f)),
				hash21(p + cgv::vec2(5.0f, 29.0f))
			);
			};

		const int CLUMP_GRID_X = 8;
		const int CLUMP_GRID_Z = 8;
		std::vector<cgv::vec2> clumpCenters(CLUMP_GRID_X * CLUMP_GRID_Z);

		for (int z = 0; z < CLUMP_GRID_Z; ++z) {
			for (int x = 0; x < CLUMP_GRID_X; ++x) {
				cgv::vec2 cellMin(
					(float)x / (float)CLUMP_GRID_X,
					(float)z / (float)CLUMP_GRID_Z
				);
				cgv::vec2 cellMax(
					(float)(x + 1) / (float)CLUMP_GRID_X,
					(float)(z + 1) / (float)CLUMP_GRID_Z
				);

				cgv::vec2 rnd = hash22(cgv::vec2((float)x, (float)z));
				clumpCenters[z * CLUMP_GRID_X + x] =
					cellMin + (cellMax - cellMin) * (0.6f * rnd + 0.2f);
			}
		}

		auto nearest_clump = [&](const cgv::vec2& uv, cgv::vec2& outCenter, float& outDist2) {
			outDist2 = 1e30f;
			outCenter = cgv::vec2(0.5f, 0.5f);

			for (const cgv::vec2& c : clumpCenters) {
				cgv::vec2 d = uv - c;
				float d2 = cgv::math::dot(d, d);
				if (d2 < outDist2) {
					outDist2 = d2;
					outCenter = c;
				}
			}
			};

		for (uint32_t s = 0; s < STRAND_AMOUNT; ++s) {
			cgv::vec2 rootUV(dist01(rng), dist01(rng));

			float dx = std::abs(rootUV.x() - 0.5f) * 2.0f;
			float dz = std::abs(rootUV.y() - 0.5f) * 2.0f;
			float edge = std::max(dx, dz);

			// Softer rectangular footprint
			float centerFactor = 1.0f - smoothstep01(edge);

			// Add sparse holes / breakup so it stops reading as a solid sheet
			float macroNoise = hash21(rootUV * 9.0f);
			float microNoise = hash21(rootUV * 31.0f + cgv::vec2(4.2f, 1.7f));
			float densityMask = 0.75f * centerFactor + 0.25f * macroNoise;

			// Random strand dropout
			if (densityMask < 0.18f)
				continue;
			if (microNoise < 0.10f)
				continue;

			// Clumping
			cgv::vec2 clumpCenter;
			float clumpDist2 = 0.0f;
			nearest_clump(rootUV, clumpCenter, clumpDist2);

			float clumpRadius = 0.018f + 0.02f * hash21(clumpCenter * 13.7f);
			float clumpT = std::exp(-clumpDist2 / std::max(1e-5f, clumpRadius * clumpRadius));

			// Root position with slight pull toward clump center
			cgv::vec2 rootUVClumped =
				rootUV * (1.0f - 0.18f * clumpT) + clumpCenter * (0.18f * clumpT);

			float px = (rootUVClumped.x() - 0.5f) * PATCH_WIDTH;
			float pz = (rootUVClumped.y() - 0.5f) * PATCH_DEPTH;
			cgv::vec3 rootPos(px, 0.0f, pz);

			// Much more aggressive length variation
			float lengthRnd = 0.55f + 0.9f * dist01(rng);
			float clumpLengthBoost = 0.85f + 0.45f * clumpT;
			float strandLength =
				cgv::math::lerp(LENGTH_MIN, LENGTH_MAX, centerFactor) *
				lengthRnd *
				clumpLengthBoost;

			// Some shorter broken hairs
			if (dist01(rng) < 0.12f)
				strandLength *= 0.45f + 0.35f * dist01(rng);

			// Base orientation
			float yawJitter = 1.1f * jitter(rng);
			float pitchJitter = 0.75f * jitter(rng);

			cgv::vec3 dir = BASE_DIR;
			dir = rotate_y(dir, yawJitter);
			dir = rotate_z(dir, pitchJitter);

			// Clump directional coherence
			float clumpYaw = (hash21(clumpCenter * 19.3f) - 0.5f) * 0.8f;
			float clumpPitch = (hash21(clumpCenter * 7.9f) - 0.5f) * 0.35f;
			dir = rotate_y(dir, clumpYaw * clumpT);
			dir = rotate_z(dir, clumpPitch * clumpT);
			dir = cgv::math::normalize(dir);

			// Localized flow variation instead of one global uniform flow
			cgv::vec3 flowDir = cgv::math::normalize(cgv::vec3(
				0.75f + 0.55f * (hash21(rootUV * 5.0f) - 0.5f),
				0.02f + 0.08f * hash21(rootUV * 11.0f + cgv::vec2(3.0f, 9.0f)),
				0.35f * (hash21(rootUV * 7.0f + cgv::vec2(8.0f, 2.0f)) - 0.5f)
			));

			std::vector<strand_vertex> strip;
			strip.reserve(STRAND_POINTS * 2);

			float strandPhase = dist01(rng) * 6.2831853f;
			float strandFreq = 1.5f + 2.0f * dist01(rng);
			float strandAmp = (0.03f + 0.08f * dist01(rng)) * strandLength;

			for (uint32_t i = 0; i < STRAND_POINTS; ++i) {
				float t = float(i) / float(STRAND_POINTS - 1);
				float bendAmount = t * t;
				float tipAmount = std::pow(t, 1.5f);

				// Stronger shape variation along the strand
				cgv::vec3 tangent =
					cgv::math::normalize((1.0f - bendAmount) * dir + bendAmount * flowDir);

				tangent += cgv::vec3(
					jitter(rng) * 0.12f,
					jitter(rng) * 0.06f,
					jitter(rng) * 0.12f
				);

				// Clump pull increases toward the tip
				cgv::vec3 toClump = cgv::vec3(
					(clumpCenter.x() - rootUV.x()) * PATCH_WIDTH,
					0.0f,
					(clumpCenter.y() - rootUV.y()) * PATCH_DEPTH
				);
				if (cgv::math::length(toClump) > 1e-5f) {
					cgv::vec3 clumpDir = cgv::math::normalize(toClump);
					tangent = cgv::math::normalize(
						tangent * (1.0f - 0.22f * clumpT * tipAmount) +
						clumpDir * (0.22f * clumpT * tipAmount)
					);
				}

				float lift = 0.14f * std::sin(t * 1.8f + strandPhase * 0.3f);
				float sway = std::sin(t * strandFreq + strandPhase) * strandAmp * tipAmount;

				cgv::vec3 sideVec = cgv::math::normalize(cgv::math::cross(tangent, cgv::vec3(0.0f, 1.0f, 0.0f)));
				if (cgv::math::length(sideVec) < 1e-5f)
					sideVec = cgv::vec3(1.0f, 0.0f, 0.0f);

				cgv::vec3 p = rootPos
					+ dir * (strandLength * t * 0.50f)
					+ flowDir * (strandLength * t * t * 0.70f)
					+ cgv::vec3(0.0f, lift * strandLength + t * strandLength * 0.12f, 0.0f)
					+ sideVec * sway;

				strand_vertex vL{};
				vL.center = p;
				vL.tan = cgv::math::normalize(tangent);
				vL.rootUV = rootUV;
				vL.VAlong = t;
				vL.side = -1.0f;

				strand_vertex vR = vL;
				vR.side = 1.0f;

				strip.push_back(vL);
				strip.push_back(vR);
			}

			for (uint32_t i = 0; i < STRAND_POINTS - 1; ++i) {
				const strand_vertex& L0 = strip[2 * i + 0];
				const strand_vertex& R0 = strip[2 * i + 1];
				const strand_vertex& L1 = strip[2 * (i + 1) + 0];
				const strand_vertex& R1 = strip[2 * (i + 1) + 1];

				verts.push_back(L0);
				verts.push_back(R0);
				verts.push_back(L1);

				verts.push_back(R0);
				verts.push_back(R1);
				verts.push_back(L1);
			}
		}
	}

	void init_quad_geometry(std::vector<vertex>& verts, float half_extent = 0.5f) {
		verts.clear();
		const cgv::vec3 normal = cgv::vec3(0.0f, 1.0f, 0.0f);

		verts.push_back({ {-half_extent, 0.0f, -half_extent}, normal, {0.0f, 0.0f} });
		verts.push_back({ { half_extent, 0.0f, -half_extent}, normal, {half_extent * 2, 0.0f} });
		verts.push_back({ { half_extent, 0.0f,  half_extent}, normal, {half_extent * 2, half_extent * 2} });

		verts.push_back({ {-half_extent, 0.0f, -half_extent}, normal, {0.0f, 0.0f} });
		verts.push_back({ { half_extent, 0.0f,  half_extent}, normal, {half_extent * 2, half_extent * 2} });
		verts.push_back({ {-half_extent, 0.0f,  half_extent}, normal, {0.0f, half_extent * 2} });
	}

	void init_sphere_geometry(std::vector<vertex>& verts, float radius = 0.25f) {
		verts.clear();

		const uint32_t STACKS = 8;
		const uint32_t SLICES = 16;
		const float PI = 3.14159265358979323846f;

		for (uint32_t stack = 0; stack < STACKS; ++stack) {
			float v0 = float(stack) / STACKS;
			float v1 = float(stack + 1) / STACKS;

			float phi0 = PI * (v0 - 0.5f);
			float phi1 = PI * (v1 - 0.5f);

			float y0 = std::sin(phi0);
			float y1 = std::sin(phi1);

			float r0 = std::cos(phi0);
			float r1 = std::cos(phi1);

			for (uint32_t slice = 0; slice < SLICES; ++slice) {
				float u0 = float(slice) / SLICES;
				float u1 = float(slice + 1) / SLICES;

				float theta0 = 2.0f * PI * u0;
				float theta1 = 2.0f * PI * u1;

				cgv::vec3 p00 = { r0 * std::cos(theta0), y0, r0 * std::sin(theta0) };
				cgv::vec3 p10 = { r0 * std::cos(theta1), y0, r0 * std::sin(theta1) };
				cgv::vec3 p01 = { r1 * std::cos(theta0), y1, r1 * std::sin(theta0) };
				cgv::vec3 p11 = { r1 * std::cos(theta1), y1, r1 * std::sin(theta1) };

				verts.push_back({ radius * p00, p00, cgv::vec2(0.0f, 0.0f) });
				verts.push_back({ radius * p11, p11, cgv::vec2(0.0f, 0.0f) });
				verts.push_back({ radius * p10, p10, cgv::vec2(0.0f, 0.0f) });

				verts.push_back({ radius * p00, p00, cgv::vec2(0.0f, 0.0f) });
				verts.push_back({ radius * p01, p01, cgv::vec2(0.0f, 0.0f) });
				verts.push_back({ radius * p11, p11, cgv::vec2(0.0f, 0.0f) });
			}
		}
	}

	void clear_dom_texture_array(cgv::render::context& ctx) {
		const float zero = 0.0f;
		glClearTexImage(dom_tau_array_gl, 0, GL_RED, GL_FLOAT, &zero);
	}
};

cgv::base::object_registration<bssrdf> bssrdf_registration("bssrdf");