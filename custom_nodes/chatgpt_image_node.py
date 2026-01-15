"""
ChatGPT Image Generation Node for ComfyUI
2枚の画像を入力し、ChatGPT API経由で画像生成を行うノード

使用方法:
1. API_KEYを環境変数 OPENAI_API_KEY に設定するか、ノードに直接入力
2. Image1 (ポーズ参照) と Image2 (外見参照) を接続
3. プロンプトテンプレートを必要に応じて編集
"""

import torch
import numpy as np
from PIL import Image
import io
import base64
import os

# OpenAI APIは後で設定（インストールされていなくてもノードは読み込める）
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def _get_attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _format_image_response_meta(response, fallback_size=None, fallback_quality=None):
    size = _get_attr(response, "size", None) or fallback_size
    quality = _get_attr(response, "quality", None) or fallback_quality
    usage = _get_attr(response, "usage", None)

    parts = []
    if size:
        parts.append(f"size={size}")
    if quality:
        parts.append(f"quality={quality}")
    if usage:
        input_tokens = _get_attr(usage, "input_tokens", None)
        output_tokens = _get_attr(usage, "output_tokens", None)
        details = _get_attr(usage, "input_tokens_details", None)
        text_tokens = _get_attr(details, "text_tokens", None) if details else None
        image_tokens = _get_attr(details, "image_tokens", None) if details else None
        if input_tokens is not None:
            parts.append(f"in_tokens={input_tokens}")
        if text_tokens is not None or image_tokens is not None:
            parts.append(f"in_text={text_tokens} in_image={image_tokens}")
        if output_tokens is not None:
            parts.append(f"out_tokens={output_tokens}")

    return " ".join(parts) if parts else "meta=unknown"


class ChatGPTPoseTransfer:
    """
    2枚の画像を入力し、gpt-image-1.5でポーズ転写画像を生成
    
    処理フロー:
        1. GPT-4o Visionで2枚の画像を分析
        2. gpt-image-1.5で最終画像を生成
    
    モデル切替:
        環境変数 OPENAI_IMAGE_MODEL を設定すると、そのモデルが使われます。
        例: export OPENAI_IMAGE_MODEL=gpt-image-2
        未設定時は gpt-image-1.5 がデフォルト。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_pose": ("IMAGE",),  # ポーズ参照用画像
                "image_appearance": ("IMAGE",),  # 外見参照用画像
                "prompt_template": ("STRING", {
                    "multiline": True,
                    "default": """[MASTER PROMPT — ABSOLUTE VISIBLE-REGION MAPPING (ONE SHOT / UNIVERSAL)]

Image 1 defines the entire reality of the shot.
Treat Image 1 as the finished photograph: its camera, framing, crop, visible body region, orientation, and perspective are absolute and must not change.

TASK:
Replace the person visible in Image 1 with the character from Image 2,
by mapping Image 2's full-body structure onto ONLY the body region that is visible in Image 1.

ABSOLUTE CAMERA & FRAMING LOCK (Image 1):
- Same camera angle, lens/FOV, perspective strength, horizon line, vanishing points.
- Same crop, zoom level, subject scale, placement, and orientation in frame.
- Do not expand, recenter, reframe, or reveal unseen areas.
- The final image must be indistinguishable from Image 1 in terms of shot and composition.

VISIBLE-REGION RULE (UNIVERSAL):
- Only the body parts that are visible in Image 1 may appear in the final image.
- The visible region may be any subset of the body (front, side, back, partial, fragmented, cropped).
- Do not assume full body, upper body, lower body, or any named region.
- Determine the visible body area purely from Image 1's framing and crop.
- Extract the corresponding anatomical region from Image 2's full-body structure and map it precisely into that space.

POSE & GEOMETRY SOURCE (Image 1):
- Use Image 1 exclusively for the pose, orientation, direction, balance, center of gravity, and spatial geometry of the visible region.
- Match the exact angles, rotations, depth, and foreshortening present in Image 1.
- Treat Image 1 as a geometric template for the visible portion only.

IDENTITY & BODY SOURCE (Image 2 ONLY):
- Use the exact character from Image 2.
- Preserve 100% of Image 2's body volume, thickness, mass, softness, and silhouette.
- Preserve Image 2's face, facial proportions, hairstyle, clothing, and design details wherever they intersect with the visible region.
- No slimming, averaging, reinterpretation, or proportional blending.

FULL-BODY TO PARTIAL MAPPING (CORE MECHANISM):
- Image 2 provides a complete 3D body.
- Image 1 defines which portion of that body is visible.
- Project the corresponding part of Image 2's body into Image 1's visible region with correct depth, scale, and perspective.
- The unseen parts of the body remain implied and must not be shown.

REPLACEMENT PRINCIPLE (NON-NEGOTIABLE):
Do not redraw the scene.
Do not re-shoot the subject.
Keep Image 1's shot exactly as-is and replace the person only.

The result must look like:
"The same shot as Image 1, with the Image 2 character naturally occupying the exact same visible space,
regardless of which body part is shown."

QUALITY:
Single coherent render, natural deformation consistent with body mass,
clean anatomy, no compositing seams or artifacts.
Output exactly one character, matching Image 1's visible framing."""
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "image_size": (["1024x1024", "1792x1024", "1024x1792"],),
                # qualityは low/medium/high/auto（デフォルトはlow=安価）
                "quality": (["low", "medium", "high", "auto"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("generated_image", "response_text",)
    FUNCTION = "generate"
    CATEGORY = "Yu Custom/ChatGPT"

    def tensor_to_base64(self, tensor):
        """ComfyUI IMAGE tensor → Base64文字列"""
        # tensor shape: [batch, height, width, channels]
        if tensor.dim() == 4:
            tensor = tensor[0]  # 最初の画像を取得
        
        # 値を0-1にクリップしてからuint8に変換
        img_np = np.clip(tensor.cpu().numpy(), 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def base64_to_tensor(self, b64_string):
        """Base64 → ComfyUI IMAGE tensor"""
        img_data = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W, C]

    def tensor_to_bytes(self, tensor):
        """ComfyUI IMAGE tensor → io.BytesIO (for images.edit API)"""
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        img_np = np.clip(tensor.cpu().numpy(), 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        buffer.name = "image.png"  # OpenAI API requires filename
        return buffer

    def generate(self, image_pose, image_appearance, prompt_template, api_key, image_size, quality):
        """メイン処理"""
        
        # APIキー取得（入力 or 環境変数）
        key = api_key if api_key else os.environ.get("OPENAI_API_KEY", "")
        
        if not key:
            # APIキーがない場合: ダミー画像を返す
            dummy = torch.zeros(1, 512, 512, 3)
            return (dummy, "ERROR: APIキーが設定されていません。環境変数 OPENAI_API_KEY を設定するか、ノードに入力してください。")
        
        if not OPENAI_AVAILABLE:
            dummy = torch.zeros(1, 512, 512, 3)
            return (dummy, "ERROR: openaiライブラリがインストールされていません。pip install openai を実行してください。")
        
        try:
            client = openai.OpenAI(api_key=key)
            
            # モデルは環境変数で切替可能（デフォルト: gpt-image-1.5）
            model_name = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
            
            # プロンプトにImage 1/Image 2の参照を追加
            full_prompt = f"""Image 1: pose + camera reference only.
Image 2: appearance/identity reference only.

{prompt_template}"""
            
            # 画像をバイナリ形式で準備（images.editは file-like objects を受け取る）
            pose_bytes = self.tensor_to_bytes(image_pose)
            appearance_bytes = self.tensor_to_bytes(image_appearance)
            
            # images.edit で2枚の参照画像を渡す（gpt-image-1.5は最大16枚まで）
            response = client.images.edit(
                model=model_name,
                image=[pose_bytes, appearance_bytes],
                prompt=full_prompt,
                size=image_size,
                quality=quality,
            )
            
            # 生成画像を取得
            result_data = response.data[0]
            
            if hasattr(result_data, 'b64_json') and result_data.b64_json:
                img_data = base64.b64decode(result_data.b64_json)
            elif hasattr(result_data, 'url') and result_data.url:
                import requests
                img_data = requests.get(result_data.url).content
            else:
                raise RuntimeError("No image data in response")
            
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            output_tensor = torch.from_numpy(img_np).unsqueeze(0)
            
            meta = _format_image_response_meta(response, image_size, quality)
            return (output_tensor, f"model={model_name} {meta} (images.edit, 2 refs)")
            
        except Exception as e:
            raise RuntimeError(f"OpenAI Image API error: {e}")


class ChatGPTPoseTransferEdit:
    """
    2枚の画像を gpt-image-1.5 の images.edit に渡して生成するポーズ転写ノード
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_pose": ("IMAGE",),
                "image_appearance": ("IMAGE",),
                "prompt_template": ("STRING", {
                    "multiline": True,
                    "default": """[MASTER PROMPT — ABSOLUTE VISIBLE-REGION MAPPING (ONE SHOT / UNIVERSAL)]

Image 1 defines the entire reality of the shot.
Treat Image 1 as the finished photograph: its camera, framing, crop, visible body region, orientation, and perspective are absolute and must not change.

TASK:
Replace the person visible in Image 1 with the character from Image 2,
by mapping Image 2's full-body structure onto ONLY the body region that is visible in Image 1.

ABSOLUTE CAMERA & FRAMING LOCK (Image 1):
- Same camera angle, lens/FOV, perspective strength, horizon line, vanishing points.
- Same crop, zoom level, subject scale, placement, and orientation in frame.
- Do not expand, recenter, reframe, or reveal unseen areas.
- The final image must be indistinguishable from Image 1 in terms of shot and composition.

VISIBLE-REGION RULE (UNIVERSAL):
- Only the body parts that are visible in Image 1 may appear in the final image.
- The visible region may be any subset of the body (front, side, back, partial, fragmented, cropped).
- Do not assume full body, upper body, lower body, or any named region.
- Determine the visible body area purely from Image 1's framing and crop.
- Extract the corresponding anatomical region from Image 2's full-body structure and map it precisely into that space.

POSE & GEOMETRY SOURCE (Image 1):
- Use Image 1 exclusively for the pose, orientation, direction, balance, center of gravity, and spatial geometry of the visible region.
- Match the exact angles, rotations, depth, and foreshortening present in Image 1.
- Treat Image 1 as a geometric template for the visible portion only.

IDENTITY & BODY SOURCE (Image 2 ONLY):
- Use the exact character from Image 2.
- Preserve 100% of Image 2's body volume, thickness, mass, softness, and silhouette.
- Preserve Image 2's face, facial proportions, hairstyle, clothing, and design details wherever they intersect with the visible region.
- No slimming, averaging, reinterpretation, or proportional blending.

FULL-BODY TO PARTIAL MAPPING (CORE MECHANISM):
- Image 2 provides a complete 3D body.
- Image 1 defines which portion of that body is visible.
- Project the corresponding part of Image 2's body into Image 1's visible region with correct depth, scale, and perspective.
- The unseen parts of the body remain implied and must not be shown.

REPLACEMENT PRINCIPLE (NON-NEGOTIABLE):
Do not redraw the scene.
Do not re-shoot the subject.
Keep Image 1's shot exactly as-is and replace the person only.

The result must look like:
"The same shot as Image 1, with the Image 2 character naturally occupying the exact same visible space,
regardless of which body part is shown."

QUALITY:
Single coherent render, natural deformation consistent with body mass,
clean anatomy, no compositing seams or artifacts.
Output exactly one character, matching Image 1's visible framing."""
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "image_size": (["1024x1024", "1792x1024", "1024x1792"],),
                "quality": (["low", "medium", "high", "auto"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("generated_image", "response_text",)
    FUNCTION = "generate"
    CATEGORY = "Yu Custom/ChatGPT"

    def tensor_to_png_bytes(self, tensor):
        """ComfyUI IMAGE tensor → PNG bytes (ファイル名付きBytesIO)"""
        if tensor.dim() == 4:
            tensor = tensor[0]
        img_np = np.clip(tensor.cpu().numpy(), 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        buffer.name = "input.png"
        return buffer

    def base64_to_tensor(self, b64_string):
        """Base64 → ComfyUI IMAGE tensor"""
        img_data = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    def generate(self, image_pose, image_appearance, prompt_template, api_key, image_size, quality):
        key = api_key if api_key else os.environ.get("OPENAI_API_KEY", "")

        if not key:
            dummy = torch.zeros(1, 512, 512, 3)
            return (dummy, "ERROR: APIキーが設定されていません。環境変数 OPENAI_API_KEY を設定するか、ノードに入力してください。")

        if not OPENAI_AVAILABLE:
            dummy = torch.zeros(1, 512, 512, 3)
            return (dummy, "ERROR: openaiライブラリがインストールされていません。pip install openai を実行してください。")

        try:
            pose_h, pose_w = int(image_pose.shape[1]), int(image_pose.shape[2])
            app_h, app_w = int(image_appearance.shape[1]), int(image_appearance.shape[2])
            img_pose = self.tensor_to_png_bytes(image_pose)
            img_appearance = self.tensor_to_png_bytes(image_appearance)

            client = openai.OpenAI(api_key=key)
            model_name = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1.5")

            response = client.images.edit(
                model=model_name,
                image=[img_pose, img_appearance],
                prompt=prompt_template,
                size=image_size,
                quality=quality,
            )

            if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
                output_tensor = self.base64_to_tensor(response.data[0].b64_json)
            else:
                import requests
                img_url = response.data[0].url
                img_response = requests.get(img_url)
                img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
                img_np = np.array(img).astype(np.float32) / 255.0
                output_tensor = torch.from_numpy(img_np).unsqueeze(0)

            meta = _format_image_response_meta(response, image_size, quality)
            return (output_tensor, f"model={model_name} {meta} in_pose={pose_w}x{pose_h} in_app={app_w}x{app_h} (images.edit)")

        except Exception as e:
            raise RuntimeError(f"OpenAI Image API error: {e}")


class ChatGPTPoseTransferEditV4(ChatGPTPoseTransferEdit):
    """
    2枚の画像を gpt-image-1.5 の images.edit に渡して生成するポーズ転写ノード（V4）
    毎回ランダムに再実行できる run_id を追加。
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = ChatGPTPoseTransferEdit.INPUT_TYPES()
        required = dict(base["required"])
        required["run_id"] = ("INT", {
            "default": 0,
            "min": 0,
            "max": 0xffffffffffffffff,
            "control_after_generate": True,
            "tooltip": "Seed-style run id for cache-busting; set to random/increment to rerun.",
        })
        return {"required": required}

    @classmethod
    def IS_CHANGED(cls, run_id=0, **kwargs):
        return run_id

    def generate(self, image_pose, image_appearance, prompt_template, api_key, image_size, quality, run_id):
        return super().generate(image_pose, image_appearance, prompt_template, api_key, image_size, quality)


class ChatGPTImg2Img:
    """
    シンプルなImage-to-Image: 1枚の画像とプロンプトでgpt-image-1.5による画像編集
    
    モデル切替:
        環境変数 OPENAI_IMAGE_MODEL を設定すると、そのモデルが使われます。
        例: export OPENAI_IMAGE_MODEL=gpt-image-2
        未設定時は gpt-image-1.5 がデフォルト。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Transform this image into a watercolor painting style"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "size": (["1024x1024", "1536x1024", "1024x1536", "auto"],),
                "run_id": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Seed-style run id for cache-busting; set to random/increment to rerun each time.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, run_id=0, **kwargs):
        return run_id

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("generated_image", "status",)
    FUNCTION = "generate"
    CATEGORY = "Yu Custom/ChatGPT"

    def tensor_to_png_bytes(self, tensor):
        """ComfyUI IMAGE tensor → PNG bytes (ファイル名付きBytesIO)"""
        if tensor.dim() == 4:
            tensor = tensor[0]
        # 値を0-1にクリップしてからuint8に変換
        img_np = np.clip(tensor.cpu().numpy(), 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        # OpenAI v1 APIはファイル名付きのfile-likeオブジェクトを期待
        buffer.name = "input.png"
        return buffer

    def base64_to_tensor(self, b64_string):
        """Base64 → ComfyUI IMAGE tensor"""
        img_data = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    def generate(self, image, prompt, api_key, size, run_id=0):
        """メイン処理: gpt-image-1.5でImage-to-Image（環境変数で切替可）"""
        
        key = api_key if api_key else os.environ.get("OPENAI_API_KEY", "")
        
        if not key:
            raise RuntimeError("APIキーが設定されていません")
        
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openaiライブラリがインストールされていません。pip install openai を実行してください。")
        
        try:
            # 画像をPNGバイト列（ファイル名付き）に変換
            img_file = self.tensor_to_png_bytes(image)
            
            client = openai.OpenAI(api_key=key)
            
            # 環境変数 OPENAI_IMAGE_MODEL があればそれを使う、なければ gpt-image-1.5
            model_name = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
            response = client.images.edit(
                model=model_name,
                image=img_file,  # ファイル名付きBytesIO
                prompt=prompt,
                size=size if size != "auto" else "1024x1024",
            )
            
            # 結果を取得（URL or b64_json）
            if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
                output_tensor = self.base64_to_tensor(response.data[0].b64_json)
            else:
                # URLの場合はダウンロード
                import requests
                img_url = response.data[0].url
                img_response = requests.get(img_url)
                img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
                img_np = np.array(img).astype(np.float32) / 255.0
                output_tensor = torch.from_numpy(img_np).unsqueeze(0)
            
            meta = _format_image_response_meta(response, size, None)
            return (output_tensor, f"model={model_name} {meta} (images.edit)")
            
        except Exception as e:
            raise RuntimeError(f"OpenAI Image API error: {e}")


# ノード登録
NODE_CLASS_MAPPINGS = {
    "ChatGPTPoseTransfer": ChatGPTPoseTransfer,
    "ChatGPTPoseTransferEdit": ChatGPTPoseTransferEdit,
    "ChatGPTPoseTransferEditV4": ChatGPTPoseTransferEditV4,
    "ChatGPTImg2Img": ChatGPTImg2Img,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatGPTPoseTransfer": "ChatGPT Pose Transfer 🎨",
    "ChatGPTPoseTransferEdit": "ChatGPT Pose Transfer (images.edit) 🎨",
    "ChatGPTPoseTransferEditV4": "ChatGPT Pose Transfer (images.edit v4) 🎨",
    "ChatGPTImg2Img": "ChatGPT Image-to-Image 🖼️",
}
