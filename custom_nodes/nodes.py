"""
Yu's Custom Nodes
"""

class TextConcatDisplay:
    """
    2つ以上のテキストを結合し、結合結果をノード上に表示しながらSTRINGを出力する。
    CLIPTextEncodeに直接接続可能。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
                "separator": ("STRING", {"default": ", "}),
            },
            "optional": {
                "text_input_1": ("STRING", {"forceInput": True}),
                "text_input_2": ("STRING", {"forceInput": True}),
                "text_input_3": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("combined_text", "display_text",)
    FUNCTION = "concat_and_display"
    CATEGORY = "Yu Custom/Text"
    OUTPUT_NODE = True  # This allows the node to display output
    
    def concat_and_display(self, base_prompt, separator, text_input_1=None, text_input_2=None, text_input_3=None):
        # 全てのテキストを集める
        texts = [base_prompt] if base_prompt else []
        
        if text_input_1:
            texts.append(text_input_1)
        if text_input_2:
            texts.append(text_input_2)
        if text_input_3:
            texts.append(text_input_3)
        
        # 結合
        combined = separator.join(texts)
        
        # UIに表示するためのテキスト（改行を含む形式）
        display_text = f"=== 結合結果 ({len(combined)} 文字) ===\n{combined}"
        
        # 結果を返す（displayはUI用）
        return {"ui": {"text": [combined]}, "result": (combined, display_text)}


class PromptBuilderWithTags:
    """
    ベースプロンプトと抽出タグを結合し、全てのテキストを表示する統合ノード。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "base_prompt": ("STRING", {"multiline": True, "default": ""}),
                "separator": ("STRING", {"default": ", "}),
            },
            "optional": {
                "extracted_tags": ("STRING", {"forceInput": True}),
                "additional_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "STRING",)
    RETURN_NAMES = ("conditioning", "full_prompt",)
    FUNCTION = "build_prompt"
    CATEGORY = "Yu Custom/Prompt"
    OUTPUT_NODE = True
    
    def build_prompt(self, clip, base_prompt, separator, extracted_tags=None, additional_prompt=None):
        # テキストを結合
        parts = []
        if base_prompt:
            parts.append(base_prompt)
        if extracted_tags:
            parts.append(extracted_tags)
        if additional_prompt:
            parts.append(additional_prompt)
        
        full_prompt = separator.join(parts)
        
        # CLIPでエンコード
        tokens = clip.tokenize(full_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        conditioning = [[cond, {"pooled_output": pooled}]]
        
        # UIに表示
        return {"ui": {"text": [full_prompt]}, "result": (conditioning, full_prompt)}


# 服・衣類関連のタグリスト
CLOTHING_TAGS = {
    # 上着
    "shirt", "t-shirt", "blouse", "sweater", "hoodie", "jacket", "coat",
    "cardigan", "vest", "tank_top", "crop_top", "polo_shirt", "dress_shirt",
    "blazer", "suit", "tuxedo", "kimono", "yukata", "haori",
    # 下着・インナー
    "underwear", "bra", "panties", "boxers", "briefs", "lingerie",
    "undershirt", "camisole", "slip",
    # ボトムス
    "pants", "jeans", "shorts", "skirt", "dress", "miniskirt", "long_skirt",
    "leggings", "tights", "stockings", "pantyhose", "thighhighs",
    "trousers", "slacks", "sweatpants", "cargo_pants",
    # 水着・スポーツウェア
    "swimsuit", "bikini", "swim_trunks", "one-piece_swimsuit",
    "sportswear", "gym_clothes", "athletic_wear",
    # アクセサリー・小物
    "hat", "cap", "beanie", "scarf", "tie", "bowtie", "necktie",
    "gloves", "mittens", "belt", "suspenders",
    # 靴・足元
    "shoes", "boots", "sneakers", "sandals", "heels", "high_heels",
    "socks", "ankle_socks", "knee_socks",
    # 制服・特殊衣装
    "school_uniform", "sailor_uniform", "maid", "maid_outfit", "nurse",
    "police_uniform", "military_uniform", "business_suit",
    # その他
    "apron", "robe", "bathrobe", "pajamas", "nightgown",
    "costume", "cosplay", "armor", "cape", "cloak",
    "collar", "choker", "necklace", "earrings", "bracelet",
    # 状態を示すタグ
    "clothed", "dressed", "wearing", "outfit", "clothing", "clothes",
    "partially_clothed", "partially_dressed",
}


class ClothingTagFilter:
    """
    1つのトグルで服を脱がせる。
    オン: 服タグを削除 + 裸タグを追加
    オフ: 何もしない（タグをそのまま通す）
    """
    
    # 裸タグ
    NUDE_TAGS = "nude, naked, completely nude, large penis, erection, prominent testicles, pubic hair"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": ("STRING", {"forceInput": True}),
                "【裸にする】make_nude": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "nude_tags": ("STRING", {
                    "multiline": True, 
                    "default": "nude, naked, completely nude, large penis, erection, prominent testicles, pubic hair"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_tags",)
    FUNCTION = "filter_tags"
    CATEGORY = "Yu Custom/Text"
    OUTPUT_NODE = True
    
    def filter_tags(self, tags, nude_tags="", **kwargs):
        # 日本語ラベル付きのパラメータ名を処理
        make_nude = kwargs.get("【裸にする】make_nude", True)
        
        # タグをリストに分割
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        
        if not make_nude:
            # オフ: 何もしない
            result = ", ".join(tag_list)
            return {"ui": {"text": [f"そのまま: {result}"]}, "result": (result,)}
        
        # オン: 服タグを削除 + 裸タグを追加
        filtered = []
        removed = []
        
        for tag in tag_list:
            tag_normalized = tag.lower().replace(" ", "_")
            
            # 服タグかチェック
            should_remove = False
            for clothing_tag in CLOTHING_TAGS:
                if clothing_tag in tag_normalized or tag_normalized in clothing_tag:
                    should_remove = True
                    break
            
            if should_remove:
                removed.append(tag)
            else:
                filtered.append(tag)
        
        # 裸タグを追加
        nude_tag_list = [t.strip() for t in (nude_tags or self.NUDE_TAGS).split(",") if t.strip()]
        filtered.extend(nude_tag_list)
        
        result = ", ".join(filtered)
        removed_str = ", ".join(removed)
        
        return {"ui": {"text": [f"裸化後: {result}\n\n削除: {removed_str}"]}, "result": (result,)}


class CharacterAttributeToggles:
    """
    キャラクター属性をオン/オフで追加できる。
    年齢、ヒゲ、メガネ、体型などを切り替え可能。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_tags": ("STRING", {"forceInput": True}),
            },
            "optional": {
                # 年齢
                "age_young": ("BOOLEAN", {"default": False}),
                "age_middle": ("BOOLEAN", {"default": False}),
                "age_old": ("BOOLEAN", {"default": False}),
                # ヒゲ
                "beard": ("BOOLEAN", {"default": False}),
                "stubble": ("BOOLEAN", {"default": False}),
                "mustache": ("BOOLEAN", {"default": False}),
                "clean_shaven": ("BOOLEAN", {"default": False}),
                # メガネ
                "glasses": ("BOOLEAN", {"default": False}),
                "sunglasses": ("BOOLEAN", {"default": False}),
                # 体型
                "muscular": ("BOOLEAN", {"default": False}),
                "chubby": ("BOOLEAN", {"default": False}),
                "fat": ("BOOLEAN", {"default": False}),
                # 体毛
                "hairy_chest": ("BOOLEAN", {"default": False}),
                "hairy_body": ("BOOLEAN", {"default": False}),
                # その他
                "sweaty": ("BOOLEAN", {"default": False}),
                "nude": ("BOOLEAN", {"default": False}),
                # カスタム追加タグ
                "custom_tags": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("modified_tags",)
    FUNCTION = "apply_toggles"
    CATEGORY = "Yu Custom/Prompt"
    OUTPUT_NODE = True
    
    # 各トグルに対応するタグ
    TOGGLE_TAGS = {
        "age_young": "young, 20 years old",
        "age_middle": "middle-aged, 40 years old, mature male",
        "age_old": "old man, elderly, 60 years old",
        "beard": "beard, facial hair",
        "stubble": "stubble",
        "mustache": "mustache",
        "clean_shaven": "clean-shaven, no facial hair",
        "glasses": "glasses, wearing glasses",
        "sunglasses": "sunglasses",
        "muscular": "muscular, muscles",
        "chubby": "chubby, thick",
        "fat": "fat, obese, big belly",
        "hairy_chest": "hairy chest",
        "hairy_body": "hairy body, body hair",
        "sweaty": "sweaty, sweat",
        "nude": "nude, naked, completely nude",
    }
    
    def apply_toggles(self, input_tags, custom_tags="", **kwargs):
        # 入力タグをリストに
        tag_list = [t.strip() for t in input_tags.split(",") if t.strip()]
        
        # 追加するタグ
        added_tags = []
        
        for toggle_name, tag_value in self.TOGGLE_TAGS.items():
            if kwargs.get(toggle_name, False):
                added_tags.append(tag_value)
        
        # カスタムタグ
        if custom_tags:
            custom = [t.strip() for t in custom_tags.split(",") if t.strip()]
            added_tags.extend(custom)
        
        # 結合
        all_tags = tag_list + added_tags
        result = ", ".join(all_tags)
        
        return {"ui": {"text": [result]}, "result": (result,)}


class PromptCategories:
    """
    プロンプトを3つのカテゴリに分けて入力し、結合する。
    各カテゴリにラベルが表示される。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "【体型】body_type": ("STRING", {
                    "multiline": True, 
                    "default": "1boy, japanese man, chubby, muscular, hairy body, mature male, 40 years old"
                }),
                "【カメラ/雰囲気】camera_style": ("STRING", {
                    "multiline": True, 
                    "default": "(delicate outlines, soft edges:1.3), detailed eyes, soft lighting"
                }),
                "【ポーズ/状況】pose_situation": ("STRING", {
                    "multiline": True, 
                    "default": ""  # 空欄
                }),
                "input_tags": ("STRING", {"forceInput": True}),
                "separator": ("STRING", {"default": ", "}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_prompt",)
    FUNCTION = "combine_categories"
    CATEGORY = "Yu Custom/Prompt"
    OUTPUT_NODE = True
    
    def combine_categories(self, separator=", ", input_tags=None, **kwargs):
        # カテゴリ名から実際の値を取得
        body = kwargs.get("【体型】body_type", "")
        camera = kwargs.get("【カメラ/雰囲気】camera_style", "")
        pose = kwargs.get("【ポーズ/状況】pose_situation", "")
        
        # 空でない項目だけ結合
        parts = []
        if body:
            parts.append(body)
        if camera:
            parts.append(camera)
        if pose:
            parts.append(pose)
        if input_tags:
            parts.append(input_tags)
        
        result = separator.join(parts)
        
        return {"ui": {"text": [result]}, "result": (result,)}


class ClothingAdder:
    """
    服を着させるノード V2。
    オン: 卑猥タグを徹底削除 + 服タグを追加
    オフ: 何もしない
    """
    
    # 削除する卑猥タグ（徹底的に）
    NUDE_TAGS_TO_REMOVE = {
        # 裸体全般
        "nude", "naked", "completely_nude", "full_nudity", "topless", "bottomless",
        "nudity", "undressed", "unclothed", "bare", "no_clothing", "no_clothes",
        "no_panties", "no_bra", "no_underwear",
        # 乳首・胸関連
        "nipples", "nipple", "areola", "areolae", "puffy_nipples", "erect_nipples",
        "dark_areola", "exposed_nipples", "visible_nipples", "thick_nipples",
        "bare_chest", "exposed_chest", "moobs", "man_tits", "sagging_breasts",
        "hanging_breasts", "cleavage", "deep_cleavage", "sweaty_cleavage", "pecs",
        # 性器関連
        "penis", "large_penis", "large penis", "huge_penis", "giant_penis", "monster_penis",
        "erection", "erect_penis", "cock", "dick", "phallus", "shaft", "glans",
        "testicles", "balls", "scrotum", "nutsack", "hanging_balls", "huge_balls",
        "foreskin", "uncircumcised", "circumcised", "precum", "urethra",
        "genitals", "genital", "crotch", "groin", "pubic", "pubic_hair", "pubic hair",
        "pubic_region", "exposed_genitals", "visible_penis", "bulge",
        
        # 肛門・お尻関連  
        "anus", "anal", "asshole", "butthole", "gaping", "spread_anus",
        "butt", "buttocks", "bare_ass", "bare ass", "exposed_buttocks",
        
        # 卑猥な行為・状態
        "masturbation", "ejaculation", "cum", "semen", "cumshot", "facial",
        "sex", "intercourse", "penetration", "insertion", "fucking",
        "orgasm", "climax", "aroused", "arousal", "horny",
        "bdsm", "bondage", "fetish", "kink",
        
        # 露出・見せる系
        "exposed", "revealing", "uncensored", "explicit", "obscene",
        "frontal_nudity", "full_frontal", "full frontal",
        "visible_genitals", "showing_genitals",
        
        # 体液・汗関連
        "sweat", "sweaty", "dripping", "wet_body", "oily", "glistening",
        "drool", "saliva", "tears",
    }
    
    # 追加する服タグ
    DEFAULT_CLOTHING = "t-shirt, casual t-shirt, shorts, casual shorts, clothed, fully clothed, casual wear, casual outfit"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": ("STRING", {"forceInput": True}),
                "【服を着させる】add_clothing": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "clothing_tags": ("STRING", {
                    "multiline": True, 
                    "default": "t-shirt, casual t-shirt, shorts, casual shorts, clothed, fully clothed, casual wear, casual outfit"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_tags",)
    FUNCTION = "add_clothing"
    CATEGORY = "Yu Custom/Text"
    OUTPUT_NODE = True
    
    def add_clothing(self, tags, clothing_tags="", **kwargs):
        add_clothing_flag = kwargs.get("【服を着させる】add_clothing", True)
        
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        
        if not add_clothing_flag:
            result = ", ".join(tag_list)
            return {"ui": {"text": [f"そのまま: {result}"]}, "result": (result,)}
        
        # 裸タグを削除
        filtered = []
        removed = []
        
        for tag in tag_list:
            tag_normalized = tag.lower().replace(" ", "_")
            
            should_remove = False
            for nude_tag in self.NUDE_TAGS_TO_REMOVE:
                nude_normalized = nude_tag.lower().replace(" ", "_")
                if nude_normalized in tag_normalized or tag_normalized in nude_normalized:
                    should_remove = True
                    break
            
            if should_remove:
                removed.append(tag)
            else:
                filtered.append(tag)
        
        # 服タグを追加
        clothing_tag_list = [t.strip() for t in (clothing_tags or self.DEFAULT_CLOTHING).split(",") if t.strip()]
        filtered.extend(clothing_tag_list)
        
        result = ", ".join(filtered)
        removed_str = ", ".join(removed)
        
        return {"ui": {"text": [f"着衣後: {result}\n\n削除: {removed_str}"]}, "result": (result,)}


import torch
import math


class ImageSizeToolbox:
    """
    サイズ系統合ノード。
    1つのノードで様々なリサイズ・スケール機能を切り替えて使用可能。
    """
    
    MODES = [
        "Scale Up (倍率)",
        "Scale Down (半分)",
        "Size Up (最大辺)",
        "Fixed Size (幅x高さ指定)",
        "4000x6000 Portrait",
        "6000x4000 Landscape",
        "Megapixel",
        "Sharpen Edges",
    ]
    
    UPSCALE_METHODS = ["lanczos", "bicubic", "bilinear", "nearest-exact", "area"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": "Scale Up (倍率)"}),
                "upscale_method": (cls.UPSCALE_METHODS, {"default": "lanczos"}),
            },
            "optional": {
                "scale_factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "target_size": ("INT", {"default": 2048, "min": 64, "max": 16384, "step": 64}),
                "width": ("INT", {"default": 4000, "min": 64, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 6000, "min": 64, "max": 16384, "step": 1}),
                "megapixels": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "sharpen_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "keep_aspect_ratio": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "Yu Custom/Image"
    OUTPUT_NODE = False
    
    def process(self, image, mode, upscale_method, 
                scale_factor=2.0, target_size=2048, 
                width=4000, height=6000, 
                megapixels=8.0, sharpen_amount=1.0,
                keep_aspect_ratio=True):
        
        # 画像形式: (batch, height, width, channels)
        batch, h, w, c = image.shape
        
        # 処理前にチャンネルを先に持ってくる (batch, channels, height, width)
        samples = image.movedim(-1, 1)
        
        if mode == "Scale Up (倍率)":
            new_w = round(w * scale_factor)
            new_h = round(h * scale_factor)
            result = self._upscale(samples, new_w, new_h, upscale_method)
            info = f"Scale Up: {w}x{h} → {new_w}x{new_h} ({scale_factor}x)"
        
        elif mode == "Scale Down (半分)":
            new_w = round(w * 0.5)
            new_h = round(h * 0.5)
            result = self._upscale(samples, new_w, new_h, upscale_method)
            info = f"Scale Down: {w}x{h} → {new_w}x{new_h} (0.5x)"
            
        elif mode == "Size Up (最大辺)":
            # 最大辺を指定サイズにリサイズ
            if w > h:
                new_w = target_size
                new_h = round(h * target_size / w)
            else:
                new_h = target_size
                new_w = round(w * target_size / h)
            result = self._upscale(samples, new_w, new_h, upscale_method)
            info = f"Size Up: {w}x{h} → {new_w}x{new_h} (max edge {target_size})"
            
        elif mode == "Fixed Size (幅x高さ指定)":
            if keep_aspect_ratio:
                new_w, new_h = self._fit_to_box(w, h, width, height)
            else:
                new_w, new_h = width, height
            result = self._upscale(samples, new_w, new_h, upscale_method)
            info = f"Fixed Size: {w}x{h} → {new_w}x{new_h}"
            
        elif mode == "4000x6000 Portrait":
            if keep_aspect_ratio:
                new_w, new_h = self._fit_to_box(w, h, 4000, 6000)
            else:
                new_w, new_h = 4000, 6000
            result = self._upscale(samples, new_w, new_h, upscale_method)
            info = f"4000x6000 Portrait: {w}x{h} → {new_w}x{new_h}"
            
        elif mode == "6000x4000 Landscape":
            if keep_aspect_ratio:
                new_w, new_h = self._fit_to_box(w, h, 6000, 4000)
            else:
                new_w, new_h = 6000, 4000
            result = self._upscale(samples, new_w, new_h, upscale_method)
            info = f"6000x4000 Landscape: {w}x{h} → {new_w}x{new_h}"
            
        elif mode == "Megapixel":
            # 総ピクセル数を指定
            total_pixels = int(megapixels * 1000000)
            current_pixels = w * h
            scale = math.sqrt(total_pixels / current_pixels)
            new_w = round(w * scale)
            new_h = round(h * scale)
            result = self._upscale(samples, new_w, new_h, upscale_method)
            actual_mp = (new_w * new_h) / 1000000
            info = f"Megapixel: {w}x{h} → {new_w}x{new_h} ({actual_mp:.1f}MP)"
            
        elif mode == "Sharpen Edges":
            # シャープ化処理
            result = self._sharpen(samples, sharpen_amount)
            info = f"Sharpen: amount={sharpen_amount}"
            
        else:
            result = samples
            info = "No processing"
        
        # チャンネルを最後に戻す (batch, height, width, channels)
        output = result.movedim(1, -1)
        
        return (output,)
    
    def _upscale(self, samples, width, height, method):
        """画像をリサイズ"""
        import comfy.utils
        return comfy.utils.common_upscale(samples, width, height, method, "disabled")
    
    def _fit_to_box(self, orig_w, orig_h, box_w, box_h):
        """アスペクト比を維持してボックスに収める"""
        ratio_w = box_w / orig_w
        ratio_h = box_h / orig_h
        ratio = min(ratio_w, ratio_h)
        new_w = round(orig_w * ratio)
        new_h = round(orig_h * ratio)
        return new_w, new_h
    
    def _sharpen(self, samples, amount):
        """エッジをシャープ化"""
        if amount <= 0:
            return samples
        
        # Unsharp Mask の簡易実装
        # カーネルサイズを固定して高速化
        import torch.nn.functional as F
        
        # ガウシアンぼかしカーネル (3x3)
        kernel = torch.tensor([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=samples.dtype, device=samples.device) / 16.0
        kernel = kernel.view(1, 1, 3, 3).repeat(samples.shape[1], 1, 1, 1)
        
        # ぼかし画像を作成
        blurred = F.conv2d(samples, kernel, padding=1, groups=samples.shape[1])
        
        # アンシャープマスク: 元画像 + (元画像 - ぼかし) * amount
        sharpened = samples + (samples - blurred) * amount
        
        # クリッピング
        sharpened = torch.clamp(sharpened, 0, 1)
        
        return sharpened


class StylePresetSelector:
    """
    画風プリセットをドロップダウンで選択。
    選んだ画風に対応するプロンプトを自動出力。
    """
    
    PRESETS = {
        "アニメ調 (Anime)": "anime style, cel shading, vibrant colors, clean lines, 2d animation",
        "リアル (Photorealistic)": "photorealistic, highly detailed, 8k uhd, professional photography, natural lighting",
        "油絵 (Oil Painting)": "oil painting, impasto, visible brush strokes, classical art, museum quality",
        "水彩 (Watercolor)": "watercolor painting, soft colors, flowing pigments, paper texture, artistic",
        "イラスト (Illustration)": "digital illustration, detailed, artstation, concept art, professional illustration",
        "マンガ調 (Manga)": "manga style, screentone, black and white, pen strokes, japanese comic art",
        "サイバーパンク (Cyberpunk)": "cyberpunk, neon lights, futuristic, high tech low life, blade runner style",
        "ファンタジー (Fantasy)": "fantasy art, magical, ethereal lighting, mystical atmosphere, epic fantasy",
        "レトロ (Retro/Vintage)": "retro style, vintage photo, film grain, nostalgic, 1980s aesthetic",
        "ミニマル (Minimalist)": "minimalist, clean design, simple shapes, limited color palette, modern aesthetic",
        "なし (None)": "",
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_preset": (list(cls.PRESETS.keys()), {"default": "アニメ調 (Anime)"}),
            },
            "optional": {
                "input_prompt": ("STRING", {"forceInput": True}),
                "custom_style": ("STRING", {"multiline": True, "default": ""}),
                "style_strength": (["weak", "medium", "strong"], {"default": "medium"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("combined_prompt", "style_only",)
    FUNCTION = "apply_style"
    CATEGORY = "Yu Custom/Style"
    OUTPUT_NODE = True
    
    def apply_style(self, style_preset, input_prompt=None, custom_style="", style_strength="medium"):
        # プリセットからスタイルタグを取得
        style_tags = self.PRESETS.get(style_preset, "")
        
        # カスタムスタイルがあれば上書き
        if custom_style:
            style_tags = custom_style
        
        # 強度に応じて重み付け
        if style_tags and style_strength != "medium":
            if style_strength == "strong":
                style_tags = f"({style_tags}:1.3)"
            elif style_strength == "weak":
                style_tags = f"({style_tags}:0.7)"
        
        # 入力プロンプトと結合
        parts = []
        if input_prompt:
            parts.append(input_prompt)
        if style_tags:
            parts.append(style_tags)
        
        combined = ", ".join(parts)
        
        return {"ui": {"text": [f"スタイル: {style_preset}\n結果: {combined}"]}, 
                "result": (combined, style_tags)}


class QuickMaskGenerator:
    """
    色やアルファ値に基づくクイックマスク生成。
    背景色指定で即座にマスクを作成。
    """
    
    MODES = [
        "背景色除去 (Chroma Key)",
        "明るさしきい値 (Brightness Threshold)",
        "透明度 (Alpha Channel)",
        "輪郭検出 (Edge Detection)",
        "グラデーション (Gradient Mask)",
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": "背景色除去 (Chroma Key)"}),
            },
            "optional": {
                # クロマキー用
                "key_color": (["green", "blue", "white", "black", "red"], {"default": "white"}),
                "tolerance": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                # しきい値用
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                # 反転
                "invert_mask": ("BOOLEAN", {"default": False}),
                # ぼかし
                "blur_amount": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE",)
    RETURN_NAMES = ("mask", "mask_preview",)
    FUNCTION = "generate_mask"
    CATEGORY = "Yu Custom/Mask"
    OUTPUT_NODE = False
    
    def generate_mask(self, image, mode, key_color="white", tolerance=0.3, 
                     threshold=0.5, invert_mask=False, blur_amount=0):
        import torch.nn.functional as F
        
        # 画像形式: (batch, height, width, channels)
        batch, h, w, c = image.shape
        
        if mode == "背景色除去 (Chroma Key)":
            # キーカラー定義
            key_colors = {
                "green": torch.tensor([0.0, 1.0, 0.0]),
                "blue": torch.tensor([0.0, 0.0, 1.0]),
                "white": torch.tensor([1.0, 1.0, 1.0]),
                "black": torch.tensor([0.0, 0.0, 0.0]),
                "red": torch.tensor([1.0, 0.0, 0.0]),
            }
            target = key_colors[key_color].to(image.device)
            
            # 色差を計算
            diff = torch.abs(image[..., :3] - target).mean(dim=-1)
            mask = (diff > tolerance).float()
            
        elif mode == "明るさしきい値 (Brightness Threshold)":
            # 輝度を計算 (Rec. 709)
            luminance = image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722
            mask = (luminance > threshold).float()
            
        elif mode == "透明度 (Alpha Channel)":
            if c >= 4:
                mask = image[..., 3]
            else:
                # アルファがない場合は白マスク
                mask = torch.ones(batch, h, w, device=image.device)
                
        elif mode == "輪郭検出 (Edge Detection)":
            # 簡易ソーベルフィルタ
            gray = image[..., :3].mean(dim=-1, keepdim=True)
            gray = gray.permute(0, 3, 1, 2)  # (batch, 1, h, w)
            
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
            
            edge_x = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), sobel_x)
            edge_y = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), sobel_y)
            edge = torch.sqrt(edge_x ** 2 + edge_y ** 2).squeeze(1)
            mask = (edge > threshold).float()
            
        elif mode == "グラデーション (Gradient Mask)":
            # 上から下へのグラデーション
            gradient = torch.linspace(0, 1, h, device=image.device)
            gradient = gradient.view(1, h, 1).expand(batch, h, w)
            mask = (gradient > threshold).float()
        else:
            mask = torch.ones(batch, h, w, device=image.device)
        
        # 反転
        if invert_mask:
            mask = 1.0 - mask
        
        # ぼかし
        if blur_amount > 0:
            kernel_size = blur_amount * 2 + 1
            sigma = blur_amount / 3.0
            
            # ガウシアンカーネル作成
            x = torch.arange(kernel_size, dtype=mask.dtype, device=mask.device) - blur_amount
            kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = kernel_1d.view(-1, 1) * kernel_1d.view(1, -1)
            kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
            
            mask_4d = mask.unsqueeze(1)
            padding = blur_amount
            mask_padded = F.pad(mask_4d, (padding, padding, padding, padding), mode='reflect')
            mask = F.conv2d(mask_padded, kernel_2d).squeeze(1)
        
        # プレビュー用画像
        mask_preview = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        
        return (mask, mask_preview)


class BatchPromptProcessor:
    """
    複数のプロンプトを一括処理。
    リストからランダム選択や順次選択が可能。
    """
    
    MODES = [
        "最初の1つ (First)",
        "ランダム (Random)",
        "順番 (Sequential)",
        "全て結合 (Combine All)",
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts_list": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "default": "prompt 1\n---\nprompt 2\n---\nprompt 3"
                }),
                "mode": (cls.MODES, {"default": "ランダム (Random)"}),
            },
            "optional": {
                "separator": ("STRING", {"default": "---"}),
                "seed": ("STRING", {"default": "0"}),
                "index": ("STRING", {"default": "0"}),
                "prefix": ("STRING", {"multiline": True, "default": ""}),
                "suffix": ("STRING", {"multiline": True, "default": ""}),
            }
        }


    
    RETURN_TYPES = ("STRING", "INT", "INT",)
    RETURN_NAMES = ("selected_prompt", "selected_index", "total_count",)
    FUNCTION = "process_prompts"
    CATEGORY = "Yu Custom/Prompt"
    OUTPUT_NODE = True
    
    def process_prompts(self, prompts_list, mode, separator="---", seed=None, index=None,
                       prefix="", suffix="", **kwargs):
        import random
        
        # 空文字列やNoneの場合はデフォルト値を使用、STRINGからintへ変換
        try:
            seed = int(seed) if seed and seed != "" else 0
        except (ValueError, TypeError):
            seed = 0
        try:
            index = int(index) if index and index != "" else 0
        except (ValueError, TypeError):
            index = 0
        if separator is None or separator == "":
            separator = "---"
        if prefix is None:
            prefix = ""
        if suffix is None:
            suffix = ""

        

        # プロンプトを分割
        prompts = [p.strip() for p in prompts_list.split(separator) if p.strip()]
        total = len(prompts)
        
        if total == 0:
            return {"ui": {"text": ["プロンプトがありません"]}, 
                    "result": ("", 0, 0)}
        
        # モードに応じて選択
        if mode == "最初の1つ (First)":
            selected_idx = 0
            selected = prompts[0]
            
        elif mode == "ランダム (Random)":
            # seed=0 の場合は本当のランダム（毎回違う結果）
            if seed != 0:
                random.seed(seed)
            selected_idx = random.randint(0, total - 1)
            selected = prompts[selected_idx]
            
        elif mode == "順番 (Sequential)":
            selected_idx = index % total
            selected = prompts[selected_idx]
            
        elif mode == "全て結合 (Combine All)":
            selected_idx = -1
            selected = ", ".join(prompts)
            
        else:
            selected_idx = 0
            selected = prompts[0]
        
        # 前後に追加
        parts = []
        if prefix:
            parts.append(prefix)
        parts.append(selected)
        if suffix:
            parts.append(suffix)
        
        result = ", ".join(parts)
        
        display = f"モード: {mode}\n選択: [{selected_idx}] / {total}件\n結果: {result}"
        
        return {"ui": {"text": [display]}, 
                "result": (result, selected_idx, total)}


# ノード登録
NODE_CLASS_MAPPINGS = {
    "TextConcatDisplay": TextConcatDisplay,
    "PromptBuilderWithTags": PromptBuilderWithTags,
    "ClothingTagFilter": ClothingTagFilter,
    "ClothingAdder": ClothingAdder,
    "CharacterAttributeToggles": CharacterAttributeToggles,
    "PromptCategories": PromptCategories,
    "ImageSizeToolbox": ImageSizeToolbox,
    "StylePresetSelector": StylePresetSelector,
    "QuickMaskGenerator": QuickMaskGenerator,
    "BatchPromptProcessor": BatchPromptProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextConcatDisplay": "🔗 Text Concat & Display",
    "PromptBuilderWithTags": "🏷️ Prompt Builder with Tags",
    "ClothingTagFilter": "👕 Clothing Tag Filter (脱がす)",
    "ClothingAdder": "👔 Clothing Adder (着せる)",
    "CharacterAttributeToggles": "🧑 Character Attribute Toggles",
    "PromptCategories": "📝 Prompt Categories",
    "ImageSizeToolbox": "📐 Image Size Toolbox",
    "StylePresetSelector": "🎨 Style Preset Selector",
    "QuickMaskGenerator": "🎭 Quick Mask Generator",
    "BatchPromptProcessor": "📋 Batch Prompt Processor",
}

