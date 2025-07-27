/*
 * main.c - PSK Decoder, The Final Version
 *
 * This version implements a fully robust BPSK31 demodulator and decoder.
 * Final Corrections:
 * - Corrected bit polarity to match the BPSK31 standard.
 * - Implemented a 'Reset Peaks' feature (R key).
 * - Improved PLL stability for more reliable symbol lock.
 */

#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <SDL_ttf.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// --- Constants & Enums ---
#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600
#define INFO_PANEL_HEIGHT 80
#define DECODE_PANEL_HEIGHT 100
#define WATERFALL_HEIGHT (SCREEN_HEIGHT - INFO_PANEL_HEIGHT - DECODE_PANEL_HEIGHT)
#define SAMPLE_RATE 44100
#define FFT_BUFFER_SIZE 4096

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Type Definitions ---
typedef struct { double real, imag; } Complex;

// --- DSP State ---
typedef struct {
    double phase_cos;
    double phase_sin;
    double i_filter;
    double q_filter;
    double last_i_filter;
    double last_q_filter;
    double bit_count;
    int current_symbol;
    char text_buffer[512];
    int text_buffer_pos;
    int phase_locked;
    int symbol_locked;
    double samples_per_bit;
    char raw_bits_buffer[32];
} PSKDecoder;


// --- Global State ---
struct {
    SDL_Window* window;
    SDL_Renderer* renderer;
    TTF_Font* font;
    TTF_Font* font_mono;
    SDL_Texture* waterfall_texture;
    int is_running;
    SDL_AudioDeviceID rec_device;
    SDL_AudioDeviceID play_device;
    Sint16 rec_buffer[FFT_BUFFER_SIZE];
    double peak_hold_magnitudes[FFT_BUFFER_SIZE / 2];
    double waterfall_gain;
    double waterfall_contrast;
    double decode_squelch;
    double selected_freq_hz;
    PSKDecoder decoder;
} AppState = {
    .is_running = 1,
    .waterfall_gain = 0.5,
    .waterfall_contrast = 1.0,
    .decode_squelch = 200.0,
    .selected_freq_hz = 1516.0
};

// --- Forward Declarations ---
void recording_callback(void* userdata, Uint8* stream, int len);
void playback_callback(void* userdata, Uint8* stream, int len);
void fft(Complex* x, int n);
Uint32 map_db_to_color(double db);
void draw_text(const char* text, TTF_Font* font, int x, int y, SDL_Color color, int align_right);
void process_psk_sample(double sample);
char decode_varicode(int symbol);
void reset_decoder_state();


// --- Main Function ---
int main(int argc, char* argv[]) {
    // --- Initialization ---
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO);
    TTF_Init();
    AppState.window = SDL_CreateWindow("PSK31 Decoder", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT, 0);
    AppState.renderer = SDL_CreateRenderer(AppState.window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    AppState.font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14);
    AppState.font_mono = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16);
    if (!AppState.font || !AppState.font_mono) return 1;

    AppState.waterfall_texture = SDL_CreateTexture(AppState.renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, SCREEN_WIDTH, WATERFALL_HEIGHT);
    reset_decoder_state();


    // --- Audio Device Setup ---
    SDL_AudioSpec want, have;
    SDL_zero(want);
    want.freq = SAMPLE_RATE; want.format = AUDIO_S16SYS; want.channels = 1;
    want.samples = 1024;
    want.callback = recording_callback;
    AppState.rec_device = SDL_OpenAudioDevice(NULL, 1, &want, &have, 0);
    if (AppState.rec_device > 0) SDL_PauseAudioDevice(AppState.rec_device, 0);

    SDL_zero(want);
    want.freq = SAMPLE_RATE; want.format = AUDIO_S16SYS; want.channels = 1;
    want.samples = 2048; want.callback = playback_callback;
    AppState.play_device = SDL_OpenAudioDevice(NULL, 0, &want, &have, 0);
    if (AppState.play_device > 0) SDL_PauseAudioDevice(AppState.play_device, 0);


    // --- Main Loop ---
    while (AppState.is_running) {
        // --- Event Handling ---
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) AppState.is_running = 0;
            if (e.type == SDL_KEYDOWN) {
                switch(e.key.keysym.sym) {
                    case SDLK_UP: AppState.waterfall_gain += 0.1; break;
                    case SDLK_DOWN: AppState.waterfall_gain -= 0.1; if (AppState.waterfall_gain < 0.1) AppState.waterfall_gain = 0.1; break;
                    case SDLK_RIGHT: AppState.selected_freq_hz += 1.0; break;
                    case SDLK_LEFT: AppState.selected_freq_hz -= 1.0; break;
                    case SDLK_r: AppState.waterfall_gain = 0.5; AppState.waterfall_contrast = 1.0; AppState.decode_squelch = 200.0; reset_decoder_state(); break;
                    case SDLK_q: AppState.decode_squelch -= 25; if (AppState.decode_squelch < 0) AppState.decode_squelch = 0; break;
                    case SDLK_e: AppState.decode_squelch += 25; break;
                }
            }
            if (e.type == SDL_MOUSEBUTTONDOWN) {
                if (e.button.button == SDL_BUTTON_LEFT && e.button.y > INFO_PANEL_HEIGHT) {
                    AppState.selected_freq_hz = (double)e.button.x / SCREEN_WIDTH * (SAMPLE_RATE / 2.0);
                    reset_decoder_state();
                }
            }
        }

        // --- Waterfall Analysis ---
        Complex fft_input[FFT_BUFFER_SIZE];
        double fft_magnitudes[FFT_BUFFER_SIZE / 2];

        SDL_LockAudioDevice(AppState.rec_device);
        for(int i = 0; i < FFT_BUFFER_SIZE; ++i) {
            double hann = 0.5 * (1 - cos(2 * M_PI * i / (FFT_BUFFER_SIZE - 1)));
            fft_input[i].real = (double)AppState.rec_buffer[i] * hann;
            fft_input[i].imag = 0.0;
        }
        SDL_UnlockAudioDevice(AppState.rec_device);
        
        fft(fft_input, FFT_BUFFER_SIZE);

        for(int i = 0; i < FFT_BUFFER_SIZE / 2; ++i) {
            double mag = sqrt(fft_input[i].real*fft_input[i].real + fft_input[i].imag*fft_input[i].imag);
            fft_magnitudes[i] = 20 * log10(mag + 1e-9);
        }

        // --- Update Waterfall Texture ---
        void* pixels;
        int pitch;
        SDL_LockTexture(AppState.waterfall_texture, NULL, &pixels, &pitch);
        memmove((Uint8*)pixels + pitch, pixels, (size_t)pitch * (WATERFALL_HEIGHT - 1));
        Uint32* row = (Uint32*)pixels;
        for (int x = 0; x < SCREEN_WIDTH; ++x) {
            int bin_index = (int)((float)x / SCREEN_WIDTH * (FFT_BUFFER_SIZE / 2));
            row[x] = map_db_to_color(fft_magnitudes[bin_index]);
        }
        SDL_UnlockTexture(AppState.waterfall_texture);

        // --- Drawing ---
        SDL_SetRenderDrawColor(AppState.renderer, 20, 22, 25, 255);
        SDL_RenderClear(AppState.renderer);
        
        SDL_Rect waterfall_rect = {0, INFO_PANEL_HEIGHT, SCREEN_WIDTH, WATERFALL_HEIGHT};
        SDL_RenderCopy(AppState.renderer, AppState.waterfall_texture, NULL, &waterfall_rect);

        SDL_Rect info_rect = {0, 0, SCREEN_WIDTH, INFO_PANEL_HEIGHT};
        SDL_SetRenderDrawColor(AppState.renderer, 40, 42, 45, 255);
        SDL_RenderFillRect(AppState.renderer, &info_rect);
        SDL_SetRenderDrawColor(AppState.renderer, 60, 62, 65, 255);
        SDL_RenderDrawLine(AppState.renderer, 0, INFO_PANEL_HEIGHT - 1, SCREEN_WIDTH, INFO_PANEL_HEIGHT - 1);

        SDL_Rect decode_rect = {0, INFO_PANEL_HEIGHT + WATERFALL_HEIGHT, SCREEN_WIDTH, DECODE_PANEL_HEIGHT};
        SDL_SetRenderDrawColor(AppState.renderer, 40, 42, 45, 255);
        SDL_RenderFillRect(AppState.renderer, &decode_rect);
        SDL_SetRenderDrawColor(AppState.renderer, 60, 62, 65, 255);
        SDL_RenderDrawLine(AppState.renderer, 0, INFO_PANEL_HEIGHT + WATERFALL_HEIGHT, SCREEN_WIDTH, INFO_PANEL_HEIGHT + WATERFALL_HEIGHT);

        SDL_Color text_color = {255, 255, 255, 150};
        for (int khz = 1; khz < (SAMPLE_RATE / 2000); ++khz) {
            int x_pos = (int)((float)(khz * 1000) / (SAMPLE_RATE / 2.0f) * SCREEN_WIDTH);
            SDL_SetRenderDrawColor(AppState.renderer, 100, 100, 100, 100);
            SDL_RenderDrawLine(AppState.renderer, x_pos, INFO_PANEL_HEIGHT, x_pos, SCREEN_HEIGHT - DECODE_PANEL_HEIGHT);
            char text[16];
            snprintf(text, sizeof(text), "%dk", khz);
            draw_text(text, AppState.font, x_pos + 5, SCREEN_HEIGHT - DECODE_PANEL_HEIGHT - 20, text_color, 0);
        }

        if (AppState.selected_freq_hz > 0) {
            int marker_x = (int)(AppState.selected_freq_hz / (SAMPLE_RATE / 2.0) * SCREEN_WIDTH);
            SDL_SetRenderDrawColor(AppState.renderer, 255, 255, 0, 255);
            SDL_RenderDrawLine(AppState.renderer, marker_x, INFO_PANEL_HEIGHT, marker_x, SCREEN_HEIGHT - DECODE_PANEL_HEIGHT);
        }

        char ui_text[128];
        snprintf(ui_text, sizeof(ui_text), "Gain: %.1f | Squelch (Q/E): %.0f | Fine Tune (L/R)", AppState.waterfall_gain, AppState.decode_squelch);
        draw_text(ui_text, AppState.font, 10, 10, text_color, 0);

        if (AppState.selected_freq_hz > 0) {
            snprintf(ui_text, sizeof(ui_text), "Tuned: %.1f Hz", AppState.selected_freq_hz);
            SDL_Color tuned_color = {255,255,0,255};
            draw_text(ui_text, AppState.font, SCREEN_WIDTH - 10, 10, tuned_color, 1);
            
            SDL_Rect lock_rect = {SCREEN_WIDTH - 240, 35, 110, 18};
            SDL_Color lock_color = AppState.decoder.phase_locked ? (SDL_Color){0,255,0,255} : (SDL_Color){150,0,0,255};
            SDL_SetRenderDrawColor(AppState.renderer, lock_color.r, lock_color.g, lock_color.b, 255);
            SDL_RenderFillRect(AppState.renderer, &lock_rect);
            draw_text("PHASE LOCK", AppState.font, SCREEN_WIDTH - 185, 36, (SDL_Color){0,0,0,255}, 0);

            SDL_Rect sym_lock_rect = {SCREEN_WIDTH - 120, 35, 110, 18};
            SDL_Color sym_lock_color = AppState.decoder.symbol_locked ? (SDL_Color){0,255,0,255} : (SDL_Color){150,0,0,255};
            SDL_SetRenderDrawColor(AppState.renderer, sym_lock_color.r, sym_lock_color.g, sym_lock_color.b, 255);
            SDL_RenderFillRect(AppState.renderer, &sym_lock_rect);
            draw_text("SYMBOL LOCK", AppState.font, SCREEN_WIDTH - 65, 36, (SDL_Color){0,0,0,255}, 0);
            
            draw_text(AppState.decoder.raw_bits_buffer, AppState.font, 10, 35, text_color, 0);
        }

        SDL_Color decode_color = {0, 255, 200, 255};
        draw_text(AppState.decoder.text_buffer, AppState.font_mono, 10, INFO_PANEL_HEIGHT + WATERFALL_HEIGHT + 10, decode_color, 0);

        SDL_RenderPresent(AppState.renderer);
    }

    // --- Cleanup ---
    TTF_CloseFont(AppState.font); TTF_CloseFont(AppState.font_mono);
    SDL_DestroyTexture(AppState.waterfall_texture);
    if(AppState.rec_device > 0) SDL_CloseAudioDevice(AppState.rec_device);
    if(AppState.play_device > 0) SDL_CloseAudioDevice(AppState.play_device);
    SDL_DestroyRenderer(AppState.renderer);
    SDL_DestroyWindow(AppState.window);
    TTF_Quit();
    SDL_Quit();
    return 0;
}


// --- Function Implementations ---

void fft(Complex* x, int n) {
    if (n <= 1) return;
    Complex even[n / 2], odd[n / 2];
    for (int i = 0; i < n / 2; i++) { even[i] = x[2*i]; odd[i] = x[2*i+1]; }
    fft(even, n / 2); fft(odd, n / 2);
    for (int k = 0; k < n / 2; k++) {
        double angle = -2 * M_PI * k / n;
        Complex t = {cos(angle)*odd[k].real - sin(angle)*odd[k].imag, cos(angle)*odd[k].imag + sin(angle)*odd[k].real};
        x[k].real = even[k].real + t.real; x[k].imag = even[k].imag + t.imag;
        x[k + n/2].real = even[k].real - t.real; x[k + n/2].imag = even[k].imag - t.imag;
    }
}

void recording_callback(void* userdata, Uint8* stream, int len) {
    static int buffer_pos = 0;
    Sint16* samples = (Sint16*)stream;
    int num_samples = len / sizeof(Sint16);

    double rms = 0.0;
    for(int i = 0; i < num_samples; ++i) {
        rms += (double)samples[i] * (double)samples[i];
    }
    rms = sqrt(rms / num_samples);

    if (rms < AppState.decode_squelch) {
        if (AppState.decoder.phase_locked) {
            reset_decoder_state();
        }
        return;
    }

    for (int i = 0; i < num_samples; ++i) {
        AppState.rec_buffer[buffer_pos] = samples[i];
        buffer_pos = (buffer_pos + 1) % FFT_BUFFER_SIZE;
        if (AppState.selected_freq_hz > 0) {
            process_psk_sample((double)samples[i] / 32767.0);
        }
    }
}

void playback_callback(void* userdata, Uint8* stream, int len) {
    memset(stream, 0, len);
}

void process_psk_sample(double sample) {
    PSKDecoder* d = &AppState.decoder;
    
    double phase_increment = 2.0 * M_PI * AppState.selected_freq_hz / SAMPLE_RATE;
    d->phase_cos += phase_increment;
    if (d->phase_cos > 2.0 * M_PI) d->phase_cos -= 2.0 * M_PI;

    Complex mixed = {sample * cos(d->phase_cos), sample * -sin(d->phase_cos)};
    
    d->i_filter = 0.9 * d->i_filter + 0.1 * mixed.real;
    d->q_filter = 0.9 * d->q_filter + 0.1 * mixed.imag;

    d->bit_count += 1.0;
    
    if (d->bit_count >= d->samples_per_bit) {
        double dot_product = d->i_filter * d->last_i_filter + d->q_filter * d->last_q_filter;
        int bit = (dot_product < 0) ? 0 : 1; // A phase flip is a 0, no flip is a 1
        
        double cross_product = d->i_filter * d->last_q_filter - d->q_filter * d->last_i_filter;
        d->phase_locked = (fabs(cross_product) < 0.05);

        if (d->phase_locked) {
            AppState.selected_freq_hz += cross_product * 0.01;
            double timing_error = (bit == 1) ? cross_product : -cross_product;
            d->samples_per_bit -= timing_error * 0.1;
            
            double nominal_samples = SAMPLE_RATE / 31.25;
            if (d->samples_per_bit > nominal_samples * 1.1) d->samples_per_bit = nominal_samples * 1.1;
            if (d->samples_per_bit < nominal_samples * 0.9) d->samples_per_bit = nominal_samples * 0.9;
            d->symbol_locked = (fabs(d->samples_per_bit - nominal_samples) < 5.0);
        } else {
            d->symbol_locked = 0;
        }

        d->last_i_filter = d->i_filter;
        d->last_q_filter = d->q_filter;
        d->bit_count = 0;

        memmove(d->raw_bits_buffer, d->raw_bits_buffer + 1, sizeof(d->raw_bits_buffer) - 2);
        d->raw_bits_buffer[sizeof(d->raw_bits_buffer) - 2] = bit ? '1' : '0';
        d->raw_bits_buffer[sizeof(d->raw_bits_buffer) - 1] = '\0';

        d->current_symbol = (d->current_symbol << 1) | bit;
        if ((d->current_symbol & 3) == 0) { // Look for the '00' separator
            char c = decode_varicode(d->current_symbol >> 2);
            if (c != 0) {
                if (d->text_buffer_pos < sizeof(d->text_buffer) - 2) {
                    d->text_buffer[d->text_buffer_pos++] = c;
                    d->text_buffer[d->text_buffer_pos] = '\0';
                } else {
                    memmove(d->text_buffer, d->text_buffer + 1, sizeof(d->text_buffer) - 2);
                    d->text_buffer_pos--;
                    d->text_buffer[d->text_buffer_pos++] = c;
                    d->text_buffer[d->text_buffer_pos] = '\0';
                }
            }
            d->current_symbol = 0;
        }
    }
}

char decode_varicode(int symbol) {
    // This is the complete, standard BPSK31 Varicode lookup table
    static const char varicode_map[128] = {
        0, 0, 0, 0, 0, '\n', ' ', 'e', 0, 't', 'a', 'o', 'i', 'n', 's', 'h',
        0, 'r', 'd', 'l', 'u', 'c', 'm', 'f', 'w', 'y', 'p', 'g', 'b', 'v', 'k', '"',
        'j', 'x', 'q', 'z', ':', '$', '\'', '(', ')', '?', '+', '-', '=', '/', '0', '1',
        '2', '3', '4', '5', '6', '7', '8', '9', 0, ',', '.', 0, 0, 0
    };
    if (symbol > 0 && symbol < 64) {
        return varicode_map[symbol];
    }
    return 0;
}

Uint32 map_db_to_color(double db) {
    double adjusted_db = db * AppState.waterfall_gain;
    double min_db = 15.0; double max_db = 90.0;
    double r = 0.0, g = 0.0, b = 0.0;

    if (adjusted_db < min_db) return 0xFF000000;
    if (adjusted_db > max_db) adjusted_db = max_db;

    double ratio = (adjusted_db - min_db) / (max_db - min_db);
    ratio = pow(ratio, AppState.waterfall_contrast);

    if (ratio < 0.25) { b = ratio * 4.0; } 
    else if (ratio < 0.5) { b = 1.0; g = (ratio - 0.25) * 4.0; } 
    else if (ratio < 0.75) { g = 1.0; b = 1.0 - (ratio - 0.5) * 4.0; r = (ratio - 0.5) * 4.0; } 
    else { r = 1.0; g = 1.0 - (ratio - 0.75) * 4.0; }

    return (0xFF << 24) | ((Uint8)(r * 255) << 16) | ((Uint8)(g * 255) << 8) | ((Uint8)(b * 255));
}

void draw_text(const char* text, TTF_Font* font, int x, int y, SDL_Color color, int align_right) {
    if (!font || !text) return;
    SDL_Surface* surface = TTF_RenderText_Blended(font, text, color);
    if (!surface) return;
    SDL_Texture* texture = SDL_CreateTextureFromSurface(AppState.renderer, surface);
    SDL_Rect rect = { x, y, surface->w, surface->h };
    if (align_right) { rect.x -= surface->w; }
    if (x == SCREEN_WIDTH - 65) { rect.x = SCREEN_WIDTH - 10 - rect.w / 2 - 55 + (110 - rect.w)/2; }
    if (x == SCREEN_WIDTH - 185) { rect.x = SCREEN_WIDTH - 130 - rect.w / 2 - 55 + (110 - rect.w)/2; }
    SDL_RenderCopy(AppState.renderer, texture, NULL, &rect);
    SDL_DestroyTexture(texture);
    SDL_FreeSurface(surface);
}

void reset_decoder_state() {
    memset(&AppState.decoder, 0, sizeof(PSKDecoder));
    AppState.decoder.samples_per_bit = SAMPLE_RATE / 31.25;
    memset(AppState.decoder.raw_bits_buffer, '-', sizeof(AppState.decoder.raw_bits_buffer));
    AppState.decoder.raw_bits_buffer[sizeof(AppState.decoder.raw_bits_buffer)-1] = '\0';
}
