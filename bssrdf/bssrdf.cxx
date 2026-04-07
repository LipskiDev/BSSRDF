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

#include "cgv/math/fvec.h"

#include <cgv_gl/gl/gl.h>

#include <plugins/cmf_tt_gl_font/tt_gl_font_server.h>

#include <random>

#define FB_MAX_RESOLUTION 2048
#define BLUR_PASSES 5

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
			NM(0, 0) = M(0, 0);
			NM(0, 1) = M(0, 1);
			NM(0, 2) = M(0, 2);
			NM(1, 0) = M(1, 0);
			NM(1, 1) = M(1, 1);
			NM(1, 2) = M(1, 2);
			NM(2, 0) = M(2, 0);
			NM(2, 1) = M(2, 1);
			NM(2, 2) = M(2, 2);
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
			transformation_matrix = cgv::math::translate4(position) *
				cgv::math::rotate4(rotation) *
				cgv::math::scale4(scale);
		}
		void draw(render_context& rctx, shader_program& shader) {
			context& ctx = *rctx.ctx;

			shader.set_uniform(ctx, "u_model_matrix", transformation_matrix);
			shader.set_uniform(ctx, "u_model_normal_matrix", rctx.get_normal_matrix(transformation_matrix));
			shader.set_uniform(ctx, "u_lightSpaceMatrix", rctx.light_matrix * transformation_matrix);
			shader.set_uniform(ctx, "u_uv_scale", uv_scale);

			ctx.set_modelview_matrix(rctx.view_matrix * transformation_matrix);

			ctx.push_modelview_matrix();
			
			albedo_tex.enable(ctx, 5);

			va.enable(ctx);
			glDrawArrays(GL_TRIANGLES, 0, (GLsizei)verts.size());
			va.disable(ctx);

			albedo_tex.disable(ctx);

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

	view* view_ptr;
	render_context rctx;

	shader_program bssrdf_shader;
	shader_program euv_shader;
	shader_program shadow_mapping_shader;
	shader_program textured_surface;
	shader_program blur_shader;
	shader_program hair_buffer_shader;
	shader_program hair_resolve_shader;

	frame_buffer fb;
	frame_buffer all_occluders_shadow_map;
	frame_buffer external_occluders_shadow_map;
	frame_buffer blur_fb;

	frame_buffer hair_fb;
	texture hair_accum;
	texture hair_reveal;

	// Deep Opacity Maps
	frame_buffer dom_fb;
	texture dom_tau;
	shader_program dom_shader;

	// Scene buffers
	frame_buffer scene_fb;
	texture scene_texture;
	texture scene_depth;

	// Geometry buffers
	struct vertex{
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

	cgv::mat4 light_space_matrix;

	std::vector<strand_vertex> strand_vertices;
	std::vector<vertex> floor_vertices;
	std::vector<vertex> sphere_vertices;

	cgv::render::texture all_occluders_depth_map;
	cgv::render::texture external_occluders_depth_map;

	cgv::render::texture irradianceBasis[BLUR_PASSES];
	cgv::render::texture temp;

	cgv::render::texture floor_texture;

	cgv::render::texture jitter_tex;

	scene_object<strand_vertex> fur;
	scene_object<vertex> sphere;
	scene_object<vertex> floor;

	bool show_euv = false;

	directional_light light;
	float shadow_map_distance = 2.0f;
	float shadow_map_resolution = 256 * 8;

	float widthTip = 0.0035, widthRoot = 0.012;

	float sigma[5] = { 1.0, 2.0, 4.0, 8.0, 16.0 };


public:
	bssrdf() 
		: all_occluders_depth_map("flt16[D]"), 
		external_occluders_depth_map("flt16[D]"), 

		cgv::base::node("BSSRDF Demo") {
		view_ptr = nullptr;

		floor.rotation = cgv::vec3(0.0, 180.0, 0.0);
		floor.position = cgv::vec3(0.0, 0.0, 0.0);
		floor.compute_transformation();

		fur.rotation = cgv::vec3(0.0, 0.0, 0.0);
		fur.position = cgv::vec3(0.0, 0.5, 0.0);
		fur.compute_transformation();

		sphere.position = cgv::vec3(1.0, 3.0, 1.0);
		sphere.compute_transformation();

		for (int i = 0; i < BLUR_PASSES; i++) {
			irradianceBasis[i] = cgv::render::texture{ "flt16[R,G,B]" };
		}

		temp = cgv::render::texture{ "flt16[R,G,B]" };

		hair_accum = cgv::render::texture{ "flt16[R,G,B,A]" };
		hair_reveal = cgv::render::texture{ "flt16[R]" };

		scene_texture = cgv::render::texture{ "flt16[R,G,B]" };
		scene_depth = cgv::render::texture{ "flt16[D]" };
		dom_tau = cgv::render::texture{ "flt16[R]" };
	}

	std::string get_type_name(void) const { return "bssrdf"; }

	void clear(cgv::render::context& ctx) {
	}

	bool self_reflect(cgv::reflect::reflection_handler& rh) {
		return true;
	}

	void on_set(void* member_ptr) {
		update_member(member_ptr);

		if (this->is_visible()) post_redraw();
	}

	bool gui_check_value(cgv::gui::control<int>& ctrl) { return true; }

	void gui_value_changed(cgv::gui::control<int>& ctrl) {
		post_redraw();
	}

	void create_gui(void) {
		add_decorator("BSSRDF", "heading");

		add_member_control(this, "Show E(u,v)", show_euv, "check");

		add_decorator("Light Direction", "heading", "level=3");
		add_member_control(this, "X", light.direction[0], "value_slider", "min=0;max=1;step=0.01;ticks=true");
		add_member_control(this, "Y", light.direction[1], "value_slider", "min=0;max=1;step=0.01;ticks=true");
		add_member_control(this, "Z", light.direction[2], "value_slider", "min=0;max=1;step=0.01;ticks=true");

		add_decorator("Fur Geometry", "heading", "level=3");
		add_member_control(this, "Fur Tip Width", widthTip, "value_slider", "min=0.001;max=0.15;step=0.001;ticks=true");
	}

	bool init(cgv::render::context& ctx) {
		bool success = true;

		view_ptr = find_view_as_node();

		if (!bssrdf_shader.build_program(ctx, "bssrdf.glpr")) {
			std::cerr << "could not build the bssrdf shader program" << std::endl;
			exit(0);
		}

		if (!euv_shader.build_program(ctx, "euv.glpr")) {
			std::cerr << "could not build the euv shader program" << std::endl;
			exit(0);
		}

		if (!shadow_mapping_shader.build_program(ctx, "surface_depth.glpr")) {
			std::cerr << "could not build the depth map shader program" << std::endl;
			exit(0);
		}

		if (!textured_surface.build_program(ctx, "surface_textured.glpr")) {
			std::cerr << "could not build the textured surface shader program" << std::endl;
			exit(0);
		}

		if (!blur_shader.build_program(ctx, "blur.glpr")) {
			std::cerr << "could not build the blur shader program" << std::endl;
			exit(0);
		}

		if (!hair_buffer_shader.build_program(ctx, "hair_buffer.glpr")) {
			std::cerr << "could not build hair buffer shader program" << std::endl;
			exit(0);
		}

		if (!hair_resolve_shader.build_program(ctx, "hair_resolve.glpr")) {
			std::cerr << "could not build hair buffer shader program" << std::endl;
			exit(0);
		}

		if (!dom_shader.build_program(ctx, "dom.glpr")) {
			std::cerr << "could not build DOM shader program" << std::endl;
			exit(0);
		}

		init_groomed_patch_geometry(fur.verts);
		init_quad_geometry(floor.verts, 10.f);
		init_sphere_geometry(sphere.verts, 0.05);
		generate_jitter_texture(ctx);

		cgv::render::type_descriptor
			strand_vec3type = cgv::render::element_descriptor_traits<cgv::vec3>::get_type_descriptor(fur.verts[0].center);

		cgv::render::type_descriptor
			strand_vec2type = cgv::render::element_descriptor_traits<cgv::vec2>::get_type_descriptor(fur.verts[0].rootUV);

		cgv::render::type_descriptor
			strand_floattype = cgv::render::element_descriptor_traits<float>::get_type_descriptor(fur.verts[0].VAlong);

		success = fur.vb.create(ctx, &(fur.verts[0]), fur.verts.size()) && success;
		success = fur.va.create(ctx);

		success = fur.va.set_attribute_array(
			ctx, 0, strand_vec3type, fur.vb,
			offsetof(strand_vertex, center),
			fur.verts.size(), sizeof(strand_vertex)) && success;

		success = fur.va.set_attribute_array(
			ctx, 1, strand_vec3type, fur.vb,
			offsetof(strand_vertex, tan),
			fur.verts.size(), sizeof(strand_vertex)) && success;

		success = fur.va.set_attribute_array(
			ctx, 2, strand_vec2type, fur.vb,
			offsetof(strand_vertex, rootUV),
			fur.verts.size(), sizeof(strand_vertex)) && success;

		success = fur.va.set_attribute_array(
			ctx, 3, strand_floattype, fur.vb,
			offsetof(strand_vertex, VAlong),
			fur.verts.size(), sizeof(strand_vertex)) && success;

		success = fur.va.set_attribute_array(
			ctx, 4, strand_floattype, fur.vb,
			offsetof(strand_vertex, side),
			fur.verts.size(), sizeof(strand_vertex)) && success;


		cgv::render::type_descriptor
			vec3type = cgv::render::element_descriptor_traits<cgv::vec3>::get_type_descriptor(floor.verts[0].pos);

		cgv::render::type_descriptor
			vec2type = cgv::render::element_descriptor_traits<cgv::vec2>::get_type_descriptor(floor.verts[0].tcoord);

		success = floor.vb.create(ctx, &(floor.verts[0]), floor.verts.size()) && success;
		success = floor.va.create(ctx);

		success = floor.va.set_attribute_array(
			ctx, 0, vec3type, floor.vb,
			0,
			floor.verts.size(), sizeof(vertex)) && success;

		success = floor.va.set_attribute_array(
			ctx, 1, vec3type, floor.vb,
			sizeof(cgv::vec3),
			floor.verts.size(), sizeof(vertex)) && success;

		success = floor.va.set_attribute_array(
			ctx, 2, vec2type, floor.vb,
			sizeof(cgv::vec3) * 2,
			floor.verts.size(), sizeof(vertex)) && success;

		success = sphere.vb.create(ctx, &(sphere.verts[0]), sphere.verts.size()) && success;
		success = sphere.va.create(ctx);

		success = sphere.va.set_attribute_array(
			ctx, 0, vec3type, sphere.vb,
			0,
			sphere.verts.size(), sizeof(vertex)) && success;

		success = sphere.va.set_attribute_array(
			ctx, 1, vec3type, sphere.vb,
			sizeof(cgv::vec3),
			sphere.verts.size(), sizeof(vertex)) && success;

		success = sphere.va.set_attribute_array(
			ctx, 2, vec2type, sphere.vb,
			sizeof(cgv::vec3) * 2,
			sphere.verts.size(), sizeof(vertex)) && success;

		light.direction = cgv::vec3(0.53, 1.0, 0.56 );
		light.color = cgv::vec3(1.0, 1.0, 1.0);

		for (int i = 0; i < BLUR_PASSES; i++) {
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

		dom_tau.create(ctx, TextureType::TT_2D, shadow_map_resolution, shadow_map_resolution);
		dom_tau.set_wrap_s(TW_CLAMP_TO_BORDER);
		dom_tau.set_wrap_t(TW_CLAMP_TO_BORDER);
		dom_tau.set_min_filter(TF_NEAREST);
		dom_tau.set_mag_filter(TF_NEAREST);
		dom_tau.set_border_color(1.0f, 1.0f, 1.0f, 1.0f);

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
		success &= dom_fb.attach(ctx, dom_tau, 0, 0);

		success &= blur_fb.create(ctx, 512, 512);
		
		read_texture(ctx, floor.albedo_tex, "C:/dev/BSSRDF/bssrdf/res/floor_albedo.png");
		read_texture(ctx, sphere.albedo_tex, "C:/dev/BSSRDF/bssrdf/res/floor_albedo.png");
		read_texture(ctx, fur.albedo_tex, "C:/dev/BSSRDF/bssrdf/res/fur_color.png");

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

	void init_frame(cgv::render::context& ctx) {
	}

	void draw(cgv::render::context& ctx) {
		if (!view_ptr) return;

		auto set_pass_state = [&](int w, int h) {
			glViewport(0, 0, w, h);
			glDisable(GL_SCISSOR_TEST);                 // or glEnable + glScissor(0,0,w,h)
			glScissor(0, 0, w, h);                      // harmless even if scissor disabled
			};

		all_occluders_depth_map.set_compare_mode(true);
		external_occluders_depth_map.set_compare_mode(true);

		cgv::vec3 eye_pos = cgv::vec3(view_ptr->get_eye());
		cgv::vec3 light_direction = normalize(light.direction);
		cgv::mat4 light_projection = cgv::math::ortho4(-3.0f, 3.0f, -3.0f, 3.0f, 0.1f, 10.0f);
		cgv::mat4 light_view = cgv::math::look_at4(5.0f * light_direction, cgv::vec3(1.0f, 1.0f, 1.0f), cgv::vec3(0.0f, 1.0f, 0.0f));

		cgv::mat4 light_matrix = light_projection * light_view;

		rctx.store_view();
		rctx.light_matrix = light_matrix;

		ctx.push_window_transformation_array();
		ctx.set_viewport(cgv::ivec4(0, 0, shadow_map_resolution, shadow_map_resolution));

		// render shadowmap 
		all_occluders_shadow_map.enable(ctx);
		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
		glDisable(GL_CULL_FACE);

		//shadow map opengl settings
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

		external_occluders_shadow_map.enable(ctx);
		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
		glDisable(GL_CULL_FACE);

		//shadow map opengl settings
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

		// DOM pass
		dom_shader.enable(ctx);
		dom_fb.enable(ctx, 0);

		ctx.push_window_transformation_array();
		ctx.set_viewport(cgv::ivec4(0, 0, shadow_map_resolution, shadow_map_resolution));

		glDisable(GL_CULL_FACE);
		glDisable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);

		glEnable(GL_BLEND);
		glBlendEquation(GL_FUNC_ADD);
		glBlendFunc(GL_ONE, GL_ONE);

		const float zero_dom[4] = { 0, 0, 0, 0 };
		glClearBufferfv(GL_COLOR, 0, zero_dom);

		dom_shader.set_uniform(ctx, "u_lightSpaceMatrix", rctx.light_matrix);
		dom_shader.set_uniform(ctx, "widthTip", widthTip);
		dom_shader.set_uniform(ctx, "widthRoot", widthRoot);

		fur.draw(rctx, dom_shader);

		dom_fb.disable(ctx);
		dom_shader.disable(ctx);

		glDisable(GL_BLEND);
		glDepthMask(GL_TRUE);

		ctx.pop_window_transformation_array();

		// render euv texture
		euv_shader.enable(ctx);

		euv_shader.set_uniform(ctx, "u_lightDirWS", light.direction);
		euv_shader.set_uniform(ctx, "u_lightRadiance", light.color);
		euv_shader.set_uniform(ctx, "u_boundaryNormalWS", cgv::vec3(0.0, 1.0, 0.0));
		euv_shader.set_uniform(ctx, "u_ambientIrradiance", cgv::vec3(0.1));
		euv_shader.set_uniform(ctx, "u_lightSpaceMatrix", rctx.light_matrix);
		euv_shader.set_uniform(ctx, "u_widthTip", widthTip);
		euv_shader.set_uniform(ctx, "u_widthRoot", widthRoot);
		external_occluders_depth_map.enable(ctx, 0);

		fb.enable(ctx, 0);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		fb.disable(ctx);

		external_occluders_depth_map.disable(ctx);
		euv_shader.disable(ctx);

		for (int i = 1; i < BLUR_PASSES; i++) {
			// Horizontal Pass
			blur(ctx, irradianceBasis[0], cgv::vec2(1.0, 0.0), sigma[i], temp);

			// Vertical Pass
			blur(ctx, temp, cgv::vec2(0.0, 1.0), sigma[i], irradianceBasis[i]);
		}

		// render floor
		scene_fb.enable(ctx, 0);
		glClearColor(1.0, 1.0, 1.0, 1.0);
		glClearDepth(1.0);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

		textured_surface.enable(ctx);
		textured_surface.set_uniform(ctx, "u_lightdir", light.direction);
		textured_surface.set_uniform(ctx, "u_lightColor", light.color);
		textured_surface.set_uniform(ctx, "u_ambient", cgv::vec3(0.2, 0.2, 0.2));
		textured_surface.set_uniform(ctx, "u_lightdir", light.direction);
		textured_surface.set_uniform(ctx, "u_lightSpaceMatrix", rctx.light_matrix);
		textured_surface.set_uniform(ctx, "u_widthTip", widthTip);
		textured_surface.set_uniform(ctx, "u_widthRoot", widthRoot);

		all_occluders_depth_map.enable(ctx, 0);
		jitter_tex.enable(ctx, 1);
		floor_texture.enable(ctx, 5);
		floor.draw(rctx, textured_surface);
		sphere.draw(rctx, textured_surface);
		ctx.pop_modelview_matrix();

		textured_surface.disable(ctx);
		scene_fb.disable(ctx);

		// render fur
		glDisable(GL_BLEND);

		hair_buffer_shader.enable(ctx);
		hair_fb.enable(ctx, 0, 1);
		glViewport(0, 0, ctx.get_width(), ctx.get_height());

		// Clear MRTs explicitly
		const float zero4[4] = { 0,0,0,0 };
		glClearBufferfv(GL_COLOR, 0, zero4);

		const float zero1[4] = { 0,0,0,0 };
		glClearBufferfv(GL_COLOR, 1, zero1);

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glDepthMask(GL_FALSE);
		glDisable(GL_CULL_FACE);

		glEnable(GL_BLEND);

		// Attachment 0: accum (dst += src)
		glBlendEquationi(0, GL_FUNC_ADD);
		glBlendFunci(0, GL_ONE, GL_ONE);

		// Attachment 1: logReveal (dst += src)
		glBlendEquationi(1, GL_FUNC_ADD);
		glBlendFunci(1, GL_ONE, GL_ONE);

		glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glColorMaski(1, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

		irradianceBasis[0].enable(ctx, 0);
		irradianceBasis[1].enable(ctx, 1);
		irradianceBasis[2].enable(ctx, 2);
		irradianceBasis[3].enable(ctx, 3);
		irradianceBasis[4].enable(ctx, 4);
		dom_tau.enable(ctx, 6);

		hair_buffer_shader.set_uniform(ctx, "u_lightSpaceMatrix", rctx.light_matrix);
		hair_buffer_shader.set_uniform(ctx, "u_scatterStrength", 1.0f);
		hair_buffer_shader.set_uniform(ctx, "u_w0", 0.35f);
		hair_buffer_shader.set_uniform(ctx, "u_w1", 0.25f);
		hair_buffer_shader.set_uniform(ctx, "u_w2", 0.18f);
		hair_buffer_shader.set_uniform(ctx, "u_w3", 0.14f);
		hair_buffer_shader.set_uniform(ctx, "u_w4", 0.08f);

		hair_buffer_shader.set_uniform(ctx, "widthTip", widthTip);
		hair_buffer_shader.set_uniform(ctx, "widthRoot", widthRoot);
		hair_buffer_shader.set_uniform(ctx, "u_lightdir", light.direction);
		hair_buffer_shader.set_uniform(ctx, "u_lightColor", light.color);
		hair_buffer_shader.set_uniform(ctx, "u_ambient", cgv::vec3(0.2, 0.2, 0.2));

		fur.draw(rctx, hair_buffer_shader);
		dom_tau.disable(ctx);
		//draw_fur_patch(rctx, hair_buffer_shader);
		glEnable(GL_DEPTH_TEST);
		hair_buffer_shader.disable(ctx);
		hair_fb.disable(ctx);

		glDepthMask(GL_TRUE);
		glDisable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		hair_resolve_shader.enable(ctx);

		hair_accum.enable(ctx, 0);
		hair_reveal.enable(ctx, 1);
		scene_texture.enable(ctx, 2);
		hair_resolve_shader.set_uniform(ctx, "u_widthTip", widthTip);
		hair_resolve_shader.set_uniform(ctx, "u_widthRoot", widthRoot);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		hair_resolve_shader.disable(ctx);
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

		src.enable(ctx);
		blur_shader.set_uniform(ctx, "u_dir", direction);
		blur_shader.set_uniform(ctx, "u_sigma", sigma);
		blur_shader.set_uniform(ctx, "u_texelSize", cgv::vec2(1.0f / float(512), 1.0f / float(512)));

		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		glEnable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glEnable(GL_SCISSOR_TEST);
		glDepthMask(GL_TRUE);

		src.disable(ctx);

		blur_shader.disable(ctx);
		blur_fb.disable(ctx);

		ctx.pop_window_transformation_array();
	}

	bool generate_jitter_texture(cgv::render::context& ctx) {
		bool success = true;

		std::mt19937 rng(42);
		std::uniform_real_distribution<float> distr(-1.0f, 1.0f);

		// xy resolution of jitter texture
		int size = 16;
		// totl number of samples is samples_u * samples_v
		int samples_u = 8;
		int samples_v = 8;

		std::vector<cgv::vec4> jitter_data(size * size * samples_u * samples_v / 2);

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				for (int k = 0; k < samples_u * samples_v / 2; k++) {

					int x, y;
					cgv::vec4 v;

					x = k % (samples_u / 2);
					y = (samples_v - 1) - k / (samples_u / 2);

					// generate points on a regular rectangular grid with dimensions samples_u x samples_v
					v[0] = (float)(x * 2 + 0.5f) / samples_u;
					v[1] = (float)(y + 0.5f) / samples_v;
					v[2] = (float)(x * 2 + 1 + 0.5f) / samples_u;
					v[3] = v[1];

					// jitter position
					v[0] += distr(rng) * (0.5f / samples_u);
					v[1] += distr(rng) * (0.5f / samples_v);
					v[2] += distr(rng) * (0.5f / samples_u);
					v[3] += distr(rng) * (0.5f / samples_v);


					// warp to disk (does not perform as well as square samples)
					cgv::vec4 d;
			
					d = v;

					// save samples as signed bytes to reduce memory requirements
					jitter_data[(k * size * size + j * size + i)] = static_cast<cgv::vec4>(127.0f * d);
				}
			}
		}

		if (jitter_tex.is_created())
			jitter_tex.destruct(ctx);

		cgv::data::data_view jitter_dv = cgv::data::data_view(new cgv::data::data_format(size, size, samples_u * samples_v / 2, cgv::type::info::TI_INT8, cgv::data::CF_RGBA), jitter_data.data());
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
		// OBJ indices: 1-based; negative allowed (-1 == last)
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
				if (!ss.fail()) {
					rawSegments.emplace_back(ia, ib);
				}
			}
		}

		if (positions.empty() || rawSegments.empty())
			return;

		// Detect global positive index bias in the OBJ.
		// Your file uses line indices [5 .. 45295] for 45291 vertices,
		// so the bias is 4.
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

		// If the positive referenced range exactly matches the number of vertices,
		// but does not start at 1, compensate.
		if (minPositiveRef != std::numeric_limits<int>::max()) {
			const int referencedSpan = maxPositiveRef - minPositiveRef + 1;
			if (referencedSpan == (int)positions.size() && minPositiveRef != 1) {
				globalBias = minPositiveRef - 1;
			}
		}

		auto resolve_index = [&](int idx1Based) -> int {
			// Handle negative OBJ indices normally
			if (idx1Based < 0) {
				int resolved = (int)positions.size() + idx1Based;
				return resolved;
			}

			// Positive indices in this file are globally biased
			int resolved = idx1Based - 1 - globalBias;
			return resolved;
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
			if ((int)adj[v].size() != 1)
				continue;
			if (visited[v])
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
			bbMin = cgv::vec3(
				std::min(bbMin.x(), rp.x()),
				std::min(bbMin.y(), rp.y()),
				std::min(bbMin.z(), rp.z())
			);
			bbMax = cgv::vec3(
				std::max(bbMax.x(), rp.x()),
				std::max(bbMax.y(), rp.y()),
				std::max(bbMax.z(), rp.z())
			);
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
		const uint32_t STRAND_POINTS = 12; // points along centerline

		const float PATCH_WIDTH = 1.6f;
		const float PATCH_DEPTH = 1.6f;

		const float LENGTH_MIN = 0.35f;
		const float LENGTH_MAX = 0.75f;

		const cgv::vec3 BASE_DIR =
			cgv::math::normalize(cgv::vec3(1.0f, 0.25f, 0.15f));

		// Each strand with N points has (N-1) ribbon segments.
		// Each segment becomes 2 triangles = 6 vertices.
		verts.clear();
		verts.reserve(STRAND_AMOUNT * (STRAND_POINTS - 1) * 6);

		std::mt19937 rng(1337);
		std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
		std::uniform_real_distribution<float> jitter(-1.0f, 1.0f);

		auto rotate_y = [](const cgv::vec3& v, float a) {
			float c = std::cos(a);
			float s = std::sin(a);
			return cgv::vec3(
				c * v.x() + s * v.z(),
				v.y(),
				-s * v.x() + c * v.z()
			);
			};

		auto rotate_z = [](const cgv::vec3& v, float a) {
			float c = std::cos(a);
			float s = std::sin(a);
			return cgv::vec3(
				c * v.x() - s * v.y(),
				s * v.x() + c * v.y(),
				v.z()
			);
			};

		auto smoothstep01 = [](float x) {
			x = std::clamp(x, 0.0f, 1.0f);
			return x * x * (3.0f - 2.0f * x);
			};

		for (uint32_t s = 0; s < STRAND_AMOUNT; ++s) {
			cgv::vec2 rootUV(dist01(rng), dist01(rng));

			// Root on rectangular patch centered at origin
			float px = (rootUV.x() - 0.5f) * PATCH_WIDTH;
			float pz = (rootUV.y() - 0.5f) * PATCH_DEPTH;
			cgv::vec3 rootPos(px, 0.0f, pz);

			// Longer in the middle, shorter near edges
			float dx = std::abs(rootUV.x() - 0.5f) * 2.0f;
			float dz = std::abs(rootUV.y() - 0.5f) * 2.0f;
			float edge = std::max(dx, dz);
			float centerFactor = 1.0f - smoothstep01(edge);

			float strandLength =
				cgv::math::lerp(LENGTH_MIN, LENGTH_MAX, centerFactor);

			float yawJitter = 0.35f * jitter(rng);
			float pitchJitter = 0.20f * jitter(rng);

			cgv::vec3 dir = BASE_DIR;
			dir = rotate_y(dir, yawJitter);
			dir = rotate_z(dir, pitchJitter);
			dir = cgv::math::normalize(dir);

			cgv::vec3 flowDir =
				cgv::math::normalize(cgv::vec3(1.0f, 0.05f, 0.0f));

			// Build the strip samples first
			std::vector<strand_vertex> strip;
			strip.reserve(STRAND_POINTS * 2);

			for (uint32_t i = 0; i < STRAND_POINTS; ++i) {
				float t = float(i) / float(STRAND_POINTS - 1);

				float bendAmount = t * t;

				cgv::vec3 tangent =
					cgv::math::normalize((1.0f - bendAmount) * dir +
						bendAmount * flowDir);

				float lift = 0.20f * std::sin(t * 1.2f);

				cgv::vec3 p = rootPos
					+ dir * (strandLength * t * 0.65f)
					+ flowDir * (strandLength * t * t * 0.55f)
					+ cgv::vec3(0.0f, lift * strandLength, 0.0f);

				strand_vertex vL{};
				vL.center = p;
				vL.tan = tangent;
				vL.rootUV = rootUV;
				vL.VAlong = t;
				vL.side = -1.0f;

				strand_vertex vR = vL;
				vR.side = 1.0f;

				strip.push_back(vL);
				strip.push_back(vR);
			}

			// Convert strip layout into explicit triangle list
			for (uint32_t i = 0; i < STRAND_POINTS - 1; ++i) {
				const strand_vertex& L0 = strip[2 * i + 0];
				const strand_vertex& R0 = strip[2 * i + 1];
				const strand_vertex& L1 = strip[2 * (i + 1) + 0];
				const strand_vertex& R1 = strip[2 * (i + 1) + 1];

				// Triangle 1
				verts.push_back(L0);
				verts.push_back(R0);
				verts.push_back(L1);

				// Triangle 2
				verts.push_back(R0);
				verts.push_back(R1);
				verts.push_back(L1);
			}
		}
	}

	void init_quad_geometry(std::vector<vertex>& verts,  float half_extent = 0.5f) {
		verts.clear();

		const cgv::vec3 normal = cgv::vec3(0.0f, 1.0f, 0.0f);

		// Triangle 1
		verts.push_back({ {-half_extent, 0.0f, -half_extent}, normal, {0.0f, 0.0f} });
		verts.push_back({ { half_extent, 0.0f, -half_extent}, normal, {half_extent * 2, 0.0f} });
		verts.push_back({ { half_extent, 0.0f,  half_extent}, normal, {half_extent * 2, half_extent * 2} });

		// Triangle 2
		verts.push_back({ {-half_extent, 0.0f, -half_extent}, normal, {0.0f, 0.0f} });
		verts.push_back({ { half_extent, 0.0f,  half_extent}, normal, {half_extent * 2, half_extent * 2} });
		verts.push_back({ {-half_extent, 0.0f,  half_extent}, normal, {0.0f, half_extent * 2} });
	}

	void init_sphere_geometry(std::vector<vertex>& verts, float radius = 0.25f) {
		sphere_vertices.clear();

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

				// Triangle 1
				verts.push_back({ radius * p00 });
				verts.push_back({ radius * p11 });
				verts.push_back({ radius * p10 });

				
				verts.push_back({ radius * p00 });
				verts.push_back({ radius * p01 });
				verts.push_back({ radius * p11 });
			}
		}
	}

	void draw_fur_patch(render_context& rctx, shader_program& shader) {
		context& ctx = *rctx.ctx;

		shader.enable(ctx);

		shader.set_uniform(ctx, "u_model_matrix", fur.transformation_matrix);
		shader.set_uniform(ctx, "u_model_normal_matrix",
			rctx.get_normal_matrix(fur.transformation_matrix));
		shader.set_uniform(ctx, "u_lightSpaceMatrix",
			rctx.light_matrix * fur.transformation_matrix);

		ctx.set_modelview_matrix(rctx.view_matrix * fur.transformation_matrix);
		ctx.push_modelview_matrix();

		fur.albedo_tex.enable(ctx, 5);
		fur.va.enable(ctx);

		const uint32_t verts_per_strand = 12 * 2; // STRAND_POINTS * 2
		const uint32_t strand_count = static_cast<uint32_t>(fur.verts.size()) / verts_per_strand;

		for (uint32_t s = 0; s < strand_count; ++s) {
			uint32_t first = s * verts_per_strand;
			glDrawArrays(GL_TRIANGLE_STRIP, first, verts_per_strand);
		}

		fur.va.disable(ctx);
		fur.albedo_tex.disable(ctx);

		ctx.pop_modelview_matrix();
		shader.disable(ctx);
	}
};

cgv::base::object_registration<bssrdf> bssrdf_registration("bssrdf");