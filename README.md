# LTXVImgToVideoInplaceDual
First last frame implementation for LTX-2

Paste to nodes_lt.py at about line 150

```
class LTXVImgToVideoInplaceDual(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVImgToVideoInplaceDual",
            display_name="LTXV Img2Vid Inplace (Dual)",
            category="conditioning/video_models",
            inputs=[
                io.Vae.Input("vae"),
                io.Image.Input("first_image", tooltip="Image to condition on the first frame."),
                io.Image.Input("last_image", tooltip="Image to condition on the last frame."),
                io.Latent.Input("latent"),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0),
                io.Boolean.Input("bypass", default=False, tooltip="Bypass the conditioning."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def _encode_image(cls, vae, image, width, height):
        if image.shape[1] != height or image.shape[2] != width:
            pixels = comfy.utils.common_upscale(
                image.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        else:
            pixels = image
        encode_pixels = pixels[:, :, :, :3]
        return vae.encode(encode_pixels)

    @classmethod
    def execute(cls, vae, first_image, last_image, latent, strength, bypass=False) -> io.NodeOutput:
        if bypass:
            return io.NodeOutput(latent)

        # Clone to avoid mutating the original latent dict or tensor
        latent = latent.copy()
        samples = latent["samples"].clone()

        _, height_scale_factor, width_scale_factor = vae.downscale_index_formula
        batch, _, latent_frames, latent_height, latent_width = samples.shape
        width = latent_width * width_scale_factor
        height = latent_height * height_scale_factor

        # Encode both images
        t_first = cls._encode_image(vae, first_image, width, height)
        t_last = cls._encode_image(vae, last_image, width, height)

        # Preserve existing noise_mask if present, otherwise create a fresh one
        if "noise_mask" in latent:
            conditioning_latent_frames_mask = latent["noise_mask"].clone()
        else:
            conditioning_latent_frames_mask = torch.ones(
                (batch, 1, latent_frames, 1, 1),
                dtype=torch.float32,
                device=samples.device,
            )

        # First frame conditioning
        samples[:, :, :t_first.shape[2]] = t_first
        conditioning_latent_frames_mask[:, :, :t_first.shape[2]] = 1.0 - strength

        # Last frame conditioning
        last_idx = latent_frames - 1
        samples[:, :, last_idx:last_idx + t_last.shape[2]] = t_last
        conditioning_latent_frames_mask[:, :, last_idx:last_idx + t_last.shape[2]] = 1.0 - strength

        return io.NodeOutput({"samples": samples, "noise_mask": conditioning_latent_frames_mask})

    generate = execute  # TODO: remove
```
