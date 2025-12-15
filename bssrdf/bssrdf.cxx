// This source code is property of the Computer Graphics and Visualization chair
// of the TU Dresden. Do not distribute! Copyright (C) CGV TU Dresden - All
// Rights Reserved
//
// The main file of the plugin. It defines a class that demonstrates how to
// register with the scene graph, drawing primitives, creating a GUI, using a
// config file and various other parts of the framework.

// Framework core
#include <cgv/base/register.h>
#include <cgv/gui/provider.h>
#include <cgv/gui/trigger.h>
#include <cgv/math/ftransform.h>
#include <cgv/render/attribute_array_binding.h>
#include <cgv/render/drawable.h>
#include <cgv/render/frame_buffer.h>
#include <cgv/render/shader_program.h>
#include <cgv/render/texture.h>
#include <cgv/render/vertex_buffer.h>

#include "cgv/math/fvec.h"

// Framework standard libraries
#include <cgv_gl/gl/gl.h>

// Framework standard plugins
#include <plugins/cmf_tt_gl_font/tt_gl_font_server.h>

#include <random>

// Some constant symbols
#define FB_MAX_RESOLUTION 2048

// The CGV framework demonstration class
class bssrdf
    : public cgv::base::base,       // This class supports reflection
      public cgv::gui::provider,    // Instances of this class provde a GUI
      public cgv::render::drawable  // Instances of this class can be rendered
{
 protected:
  ////
  // Stuff we expose via reflection

  // Whether to use wireframe mode (helps visually debugging the custom
  // tesselation task)
  bool wireframe;

  // Internal stuff we don't expose via reflection

  cgv::render::shader_program bssrdf_shader;

  // Geometry buffers
  struct vertex {
    cgv::vec3 pos;
    cgv::vec2 tcoord;
  };

  struct strand_vertex {
    cgv::vec3 center;
    cgv::vec3 tan;
    cgv::vec2 rootUV;
    float VAlong;
    float side;
  };

  std::vector<vertex> vertices;
  std::vector<strand_vertex> strand_vertices;
  std::vector<uint32_t> strand_indices;

  cgv::render::vertex_buffer vb;
  cgv::render::vertex_buffer strands_vb;
  cgv::render::attribute_array_binding vertex_array;
  cgv::render::attribute_array_binding strands_va;

  cgv::render::texture euv;

  // Flag for checking whether we have to reinit due to change in desired
  // offscreen framebuffer resolution
  bool fb_invalid;

 public:
  // Default constructor
  bssrdf() : wireframe(false), euv(cgv::render::texture{"[R,G,B]"}) {
    // Make sure the font server knows about the fonts packaged with the
    // exercise
    // cgv::scan_fonts("./data/Fonts");
  }

  // Should be overwritten to sensibly implement the cgv::base::named
  // interface
  std::string get_type_name(void) const { return "bssrdf"; }

  // Part of the cgv::base::base interface, can be implemented to make data
  // members of this class available as named properties, e.g. for use with
  // config files
  bool self_reflect(cgv::reflect::reflection_handler &rh) {
    // Reflect the properties
    return rh.reflect_member("wireframe", wireframe);
  }

  // Part of the cgv::base::base interface, should be implemented to respond to
  // write access to reflected data members of this class, e.g. from config file
  // processing or gui interaction.
  void on_set(void *member_ptr) {
    update_member(member_ptr);

    if (this->is_visible()) post_redraw();
  }

  // We use this for validating GUI input
  bool gui_check_value(cgv::gui::control<int> &ctrl) { return true; }

  // We use this for acting upon validated GUI input
  void gui_value_changed(cgv::gui::control<int> &ctrl) {
    // Redraw the scene
    post_redraw();
  }

  // Required interface for cgv::gui::provider
  void create_gui(void) {}

  // Part of the cgv::render::drawable interface, can be overwritten if there is
  // some intialization work to be done that needs a set-up and ready graphics
  // context, which usually you don't have at object construction time. Should
  // return true if the initialization was successful, false otherwise.
  bool init(cgv::render::context &ctx) {
    // Keep track of success - do it this way (instead of e.g. returning false
    // immediatly) to perform every init step even if some go wrong.
    bool success = true;

    // Init geometry buffers
    // - get a reference to the default shader, from which we're going to query
    // named
    //   locations of the vertex layout
    // cgv::render::shader_program &default_shader =
    //     ctx.ref_default_shader_program(true /* true for texture support */);

    if (!bssrdf_shader.build_program(ctx, "bssrdf.glpr")) {
      std::cerr << "could not build the bssrdf shader program" << std::endl;
      exit(0);
    }

    init_strands_geometry();

    cgv::render::type_descriptor
        strand_vec3type = cgv::render::element_descriptor_traits<
            cgv::vec3>::get_type_descriptor(strand_vertices[0].center),
        strand_vec2type = cgv::render::element_descriptor_traits<
            cgv::vec2>::get_type_descriptor(strand_vertices[0].rootUV),
        strand_floattype =
            cgv::render::element_descriptor_traits<float>::get_type_descriptor(
                strand_vertices[0].VAlong);

    success =
        strands_vb.create(ctx, &(strand_vertices[0]), strand_vertices.size()) &&
        success;

    success == strands_va.create(ctx);

    success = strands_va.set_attribute_array(
                  ctx, 0, strand_vec3type, strands_vb, 0,
                  strand_vertices.size(), sizeof(strand_vertex)) &&
              success;

    success = strands_va.set_attribute_array(
                  ctx, 1, strand_vec3type, strands_vb, sizeof(cgv::vec3),
                  strand_vertices.size(), sizeof(strand_vertex)) &&
              success;

    success = strands_va.set_attribute_array(
                  ctx, 2, strand_vec2type, strands_vb, sizeof(cgv::vec3) * 2,
                  strand_vertices.size(), sizeof(strand_vertex)) &&
              success;

    success = strands_va.set_attribute_array(
                  ctx, 3, strand_floattype, strands_vb,
                  sizeof(cgv::vec3) * 2 + sizeof(cgv::vec2),
                  strand_vertices.size(), sizeof(strand_vertex)) &&
              success;

    success = strands_va.set_attribute_array(
                  ctx, 4, strand_floattype, strands_vb,
                  sizeof(cgv::vec3) * 2 + sizeof(cgv::vec2) + sizeof(float),
                  strand_vertices.size(), sizeof(strand_vertex)) &&
              success;

    // Flag offscreen framebuffer as taken care of
    fb_invalid = false;

    // All initialization has been attempted
    return success;
  }

  // Part of the cgv::render::drawable interface, can be overwritten if there is
  // some work to be done before actually rendering a frame.
  void init_frame(cgv::render::context &ctx) {}

  // Should be overwritten to sensibly implement the cgv::render::drawable
  // interface
  void draw(cgv::render::context &ctx) {
    // Observe wireframe mode
    glPushAttrib(GL_POLYGON_BIT);
    if (wireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // Shortcut to the built-in default shader with lighting and texture support

    // Enable shader program we want to use for drawing
    bssrdf_shader.enable(ctx);

    bssrdf_shader.set_uniform(ctx, "u_lightdir", cgv::vec3(0.5, 0.5, 0.5));
    bssrdf_shader.set_uniform(ctx, "u_lightColor", cgv::vec3(1.0, 1.0, 1.0));
    bssrdf_shader.set_uniform(ctx, "u_ambient", cgv::vec3(0.2, 0.2, 0.2));

    // Set the "color" vertex attribute for all geometry drawn hereafter, except
    // if it explicitely specifies its own color data by means of an attribute
    // array. Note that this only works for shaders that define a vec4 attribute
    // named "color" in their layout specification. We want white to retain the
    // original color information in the texture.
    ctx.set_color(cgv::rgb(1.0f));

    // Draw the node's scene geometry - save current modelview matrix because we
    // have some node-internal transformations that we do not want to spill over
    // to other drawables.
    ctx.push_modelview_matrix();
    // // Account for aspect ratio of the offscreen texture
    // ctx.mul_modelview_matrix(cgv::math::rotate4(90.0, 0.0, 0.0, 0.0));

    ctx.mul_modelview_matrix(cgv::math::translate4(0.0, -0.1, 0.0));
    ctx.mul_modelview_matrix(cgv::math::rotate4(-90.0, 1.0, 0.0, 0.0));

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    draw_fur(ctx);

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //*****************************************************************/
    glPopAttrib();
    ctx.pop_modelview_matrix();

    // Disable shader program and texture
    bssrdf_shader.disable(ctx);
  }

  void init_strands_geometry(void) {
    const uint32_t STRAND_AMOUNT = 360000;
    const uint32_t STRAND_POINTS = 3;
    const float LENGTH_MIN = 0.06f;
    const float LENGTH_MAX = 0.14f;

    strand_vertices.clear();
    strand_indices.clear();

    strand_vertices.reserve(STRAND_AMOUNT * STRAND_POINTS * 2);

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_real_distribution<float> jitter(-1.0f, 1.0f);

    auto quadPosFromUV = [&](const cgv::vec2 &uv) {
      return cgv::vec3(uv.x() - 0.5f, 0.0f, uv.y() - 0.5f);
    };

    for (uint32_t s = 0; s < STRAND_AMOUNT; s++) {
      cgv::vec2 rootUV(dist01(rng), dist01(rng));

      cgv::vec3 rootPos = quadPosFromUV(rootUV);

      cgv::vec2 baseDir2D = cgv::math::normalize(cgv::vec2(1.0f, 0.25f));
      float angle = 0.25f * jitter(rng);
      float ca = std::cos(angle);
      float sa = std::sin(angle);

      cgv::vec2 dir2D(ca * baseDir2D.x() - sa * baseDir2D.y(),
                      sa * baseDir2D.x() + ca * baseDir2D.y());

      cgv::vec3 T = cgv::math::normalize(cgv::vec3(dir2D.x(), 1.0f, dir2D.y()));

      float L = cgv::math::lerp(LENGTH_MIN, LENGTH_MAX, dist01(rng));

      for (uint32_t i = 0; i < STRAND_POINTS; i++) {
        float t = float(i) / float(STRAND_POINTS - 1);

        cgv::vec3 center = rootPos + T * (L * t);

        strand_vertex vL{};
        vL.center = center;
        vL.tan = T;
        vL.rootUV = rootUV;
        vL.VAlong = t;
        vL.side = -1.0f;

        strand_vertex vR = vL;
        vR.side = 1.0f;

        strand_vertices.push_back(vL);
        strand_vertices.push_back(vR);
      }
    }
  }

  // Draw method for a custom quad
  void draw_fur(cgv::render::context &ctx) {
    strands_va.enable(ctx);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)strand_vertices.size());
    strands_va.disable(ctx);
  }
};

// Create an instance of the demo class at plugin load and register it with the
// framework
cgv::base::object_registration<bssrdf> bssrdf_registration("");

// The following could be used to register the class with the framework but NOT
// create it upon plugin load. Instead, the user can create an instance from the
// application menu. However, config files are not straight-forward to use in
// this case, which is why we go for the method above.
/*
        cgv::base::factory_registration<cgv_demo> cgv_demo_factory(
                "new/cgv_demo", // menu path
                'D',            // the shortcut - capital D means ctrl+d
                true            // whether the class is supposed to be a
   singleton
        );
*/
