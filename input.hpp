#include "updater.hpp"
#include "generator.hpp"
#include "./include/raylib.h"
#include <string> // Include for std::to_string

class MyInput
{
public:
    MyInput() = default;
    ~MyInput() = default;
    bool show_text = true;

    void processInput(Gen &gen, EulerUpdater &updater, size_t &emmit_rate)
    {

        if (show_text) {
            DrawText(("floorY : " + std::to_string(updater.m_floorY) + "; press q or w to change").c_str(), 10, 110, 20, DARKGRAY);
        }

        if (IsKeyDown(KEY_Q)) {
            updater.m_floorY += 1.0f;
        }
        if (IsKeyDown(KEY_W)) {
            updater.m_floorY -= 1.0f;
        }
        if(show_text){

            DrawText(("bounceFactor : " + std::to_string(updater.m_bounceFactor) + "; press e or r to change").c_str(), 10, 130, 20, DARKGRAY);
        }
        if (IsKeyDown(KEY_E)) {
            updater.m_bounceFactor += 0.1f;
        }
        if (IsKeyDown(KEY_R)) {
            updater.m_bounceFactor -= 0.1f;
        }
        if(show_text) {
            DrawText(("minStartCol : " + std::to_string(gen.m_minStartCol.x()) + ", " + std::to_string(gen.m_minStartCol.y()) + ", " + std::to_string(gen.m_minStartCol.z()) + "; press {a, s, d} or {z, x, c} to change").c_str(), 10, 150, 20, DARKGRAY);
        }
        if (IsKeyDown(KEY_A) && gen.m_minStartCol.x() < 255.0f) {
            gen.m_minStartCol.x() += 1.0f;
        }
        if (IsKeyDown(KEY_S) && gen.m_minStartCol.x() > 0.0f) {
            gen.m_minStartCol.x() -= 1.0f;
        }
        if (IsKeyDown(KEY_D) && gen.m_minStartCol.y() < 255.0f ) {
            gen.m_minStartCol.y() += 1.0f;
        }
        if (IsKeyDown(KEY_Z) && gen.m_minStartCol.y() > 0.0f) {
            gen.m_minStartCol.y() -= 1.0f;
        }
        if (IsKeyDown(KEY_X) && gen.m_minStartCol.z() < 255.0f) {
            gen.m_minStartCol.z() += 1.0f;
        }
        if (IsKeyDown(KEY_C) && gen.m_minStartCol.z() > 0.0f) {
            gen.m_minStartCol.z() -= 1.0f;
        }
        if(show_text)
        {
            DrawText(("maxStartCol : " + std::to_string(gen.m_maxStartCol.x()) + ", " + std::to_string(gen.m_maxStartCol.y()) + ", " + std::to_string(gen.m_maxStartCol.z()) + "; press {f, g, h} or {v, b, n} to change").c_str(), 10, 170, 20, DARKGRAY);
        }
        if (IsKeyDown(KEY_F) && gen.m_maxStartCol.x() < 255.0f) {
            gen.m_maxStartCol.x() += 1.0f;
        }
        if (IsKeyDown(KEY_G) && gen.m_maxStartCol.x() > 0.0f) {
            gen.m_maxStartCol.x() -= 1.0f;
        }
        if (IsKeyDown(KEY_H) && gen.m_maxStartCol.y() < 255.0f) {
            gen.m_maxStartCol.y() += 1.0f;
        }
        if (IsKeyDown(KEY_V)  && gen.m_maxStartCol.y() > 0.0f) {
            gen.m_maxStartCol.y() -= 1.0f;
        }
        if (IsKeyDown(KEY_B) && gen.m_maxStartCol.z() < 255.0f) {
            gen.m_maxStartCol.z() += 1.0f;
        }
        if (IsKeyDown(KEY_N) && gen.m_maxStartCol.z() > 0.0f) {
            gen.m_maxStartCol.z() -= 1.0f;
        }
        if(show_text)
        {
            DrawText(("minEndCol : " + std::to_string(gen.m_minEndCol.x()) + ", " + std::to_string(gen.m_minEndCol.y()) + ", " + std::to_string(gen.m_minEndCol.z()) + "; press {j, k, l} or {m, comma, period} to change").c_str(), 10, 190, 20, DARKGRAY);
        }
        if (IsKeyDown(KEY_J) && gen.m_minEndCol.x() < 255.0f ) {
            gen.m_minEndCol.x() += 1.0f;
        }
        if (IsKeyDown(KEY_K) && gen.m_minEndCol.x() > 0.0f) {
            gen.m_minEndCol.x() -= 1.0f;
        }
        if (IsKeyDown(KEY_L) && gen.m_minEndCol.y() < 255.0f) {
            gen.m_minEndCol.y() += 1.0f;
        }
        if (IsKeyDown(KEY_M) && gen.m_minEndCol.y() > 0.0f) {
            gen.m_minEndCol.y() -= 1.0f;
        }
        if (IsKeyDown(KEY_COMMA) && gen.m_minEndCol.z() < 255.0f) {
            gen.m_minEndCol.z() += 1.0f;
        }
        if (IsKeyDown(KEY_PERIOD) && gen.m_minEndCol.z() > 0.0f) {
            gen.m_minEndCol.z() -= 1.0f;
        }
        if(show_text)
        {
            DrawText(("maxEndCol : " + std::to_string(gen.m_maxEndCol.x()) + ", " + std::to_string(gen.m_maxEndCol.y()) + ", " + std::to_string(gen.m_maxEndCol.z()) + "; press {u, i, o} or {p, left_bracket, right_bracket} to change").c_str(), 10, 210, 20, DARKGRAY);
        }
        if (IsKeyDown(KEY_U) && gen.m_maxEndCol.x() < 255.0f) {
            gen.m_maxEndCol.x() += 1.0f;
        }
        if (IsKeyDown(KEY_I) && gen.m_maxEndCol.x() > 0.0f) {
            gen.m_maxEndCol.x() -= 1.0f;
        }
        if (IsKeyDown(KEY_O) && gen.m_maxEndCol.y() < 255.0f) {
            gen.m_maxEndCol.y() += 1.0f;
        }
        if (IsKeyDown(KEY_P) && gen.m_maxEndCol.y() > 0.0f) {
            gen.m_maxEndCol.y() -= 1.0f;
        }
        if (IsKeyDown(KEY_LEFT_BRACKET) && gen.m_maxEndCol.z() < 255.0f) {
            gen.m_maxEndCol.z() += 1.0f;
        }
        if (IsKeyDown(KEY_RIGHT_BRACKET) && gen.m_maxEndCol.z() > 0.0f) {
            gen.m_maxEndCol.z() -= 1.0f;
        }
        if(show_text)
        {
            DrawText(("minStartVel : " + std::to_string(gen.m_minStartVel.x()) + ", " + std::to_string(gen.m_minStartVel.y()) + ", " + std::to_string(gen.m_minStartVel.z()) + "; press {1, 2, 3} or {4, 5, 6} to change").c_str(), 10, 230, 20, DARKGRAY);
        }
        if (IsKeyDown(KEY_ONE)) {
            gen.m_minStartVel.x() += 1.0f;
        }
        if (IsKeyDown(KEY_TWO)) {
            gen.m_minStartVel.x() -= 1.0f;
        }
        if (IsKeyDown(KEY_THREE)) {
            gen.m_minStartVel.y() += 1.0f;
        }
        if (IsKeyDown(KEY_FOUR)) {
            gen.m_minStartVel.y() -= 1.0f;
        }
        if (IsKeyDown(KEY_FIVE)) {
            gen.m_minStartVel.z() += 1.0f;
        }
        if (IsKeyDown(KEY_SIX)) {
            gen.m_minStartVel.z() -= 1.0f;
        }
        if(show_text)
        {
            DrawText(("maxStartVel : " + std::to_string(gen.m_maxStartVel.x()) + ", " + std::to_string(gen.m_maxStartVel.y()) + ", " + std::to_string(gen.m_maxStartVel.z()) + "; press {7, 8, 9} or {0, minus, equal} to change").c_str(), 10, 250, 20, DARKGRAY);
        }
        if (IsKeyDown(KEY_SEVEN)) {
            gen.m_maxStartVel.x() += 1.0f;
        }
        if (IsKeyDown(KEY_EIGHT)) {
            gen.m_maxStartVel.x() -= 1.0f;
        }
        if (IsKeyDown(KEY_NINE)) {
            gen.m_maxStartVel.y() += 1.0f;
        }
        if (IsKeyDown(KEY_ZERO)) {
            gen.m_maxStartVel.y() -= 1.0f;
        }
        if (IsKeyDown(KEY_MINUS)) {
            gen.m_maxStartVel.z() += 1.0f;
        }
        if (IsKeyDown(KEY_EQUAL)) {
            gen.m_maxStartVel.z() -= 1.0f;
        }
        if(show_text)
        {
            DrawText(("minTime : " + std::to_string(gen.m_minTime) + "; press {tab, enter} to change").c_str(), 10, 270, 20, DARKGRAY);
        }
        if (IsKeyDown(KEY_TAB)) {
            gen.m_minTime += 1.0f;
        }
        if (IsKeyDown(KEY_ENTER)) {
            gen.m_minTime -= 1.0f;
        }
        if(show_text)
        {
            DrawText(("maxTime : " + std::to_string(gen.m_maxTime) + "; press {backspace, space} to change").c_str(), 10, 290, 20, DARKGRAY);
        }
        if (IsKeyDown(KEY_BACKSPACE)) {
            gen.m_maxTime += 1.0f;
        }
        if (IsKeyDown(KEY_SPACE)) {
            gen.m_maxTime -= 1.0f;
        }
        if(show_text)
        {
            DrawText(("maxStartPosOffset : " + std::to_string(gen.m_maxStartPosOffset.x()) + ", " + std::to_string(gen.m_maxStartPosOffset.y()) + ", " + std::to_string(gen.m_maxStartPosOffset.z()) + "; press {f1, f2, f3} or {f4, f5, f6} to change").c_str(), 10, 310, 20, DARKGRAY);
        }
        if (IsKeyDown(KEY_F1)) {
            gen.m_maxStartPosOffset.x() += 1.0f;
        }
        if (IsKeyDown(KEY_F2)) {
            gen.m_maxStartPosOffset.x() -= 1.0f;
        }
        if (IsKeyDown(KEY_F3)) {
            gen.m_maxStartPosOffset.y() += 1.0f;
        }
        if (IsKeyDown(KEY_F4)) {
            gen.m_maxStartPosOffset.y() -= 1.0f;
        }
        if (IsKeyDown(KEY_F5)) {
            gen.m_maxStartPosOffset.z() += 1.0f;
        }
        if (IsKeyDown(KEY_F6)) {
            gen.m_maxStartPosOffset.z() -= 1.0f;
        }
        if(show_text)
        {
            DrawText(("emmit_rate : " + std::to_string(emmit_rate) + "; press {up, down} to change").c_str(), 10, 330, 20, DARKGRAY);
        }
        if (IsKeyDown(KEY_UP)) {
            emmit_rate += 30;
        }
        if (IsKeyDown(KEY_DOWN) && emmit_rate > 30) {
            emmit_rate -= 30;
        }
        // if (emmit_rate < 32) emmit_rate = 0;
        if(show_text)
        {
            DrawText(("scroll your mouse wheel to zoom in/out"), 10, 350, 20, DARKGRAY);
        }
        if(show_text)
        {
            DrawText(("press ctr-h to hide/show this text"), 10, 370, 20, DARKGRAY);
        }
        
        if (IsKeyDown(KEY_LEFT_CONTROL) && IsKeyPressed(KEY_H)) {
            show_text = !show_text;
        }

        if(show_text)
        {
            DrawText(("acc_min: " + std::to_string(updater.acc_min) + " acc_max: " + std::to_string(updater.acc_max) + " press {f7, f8} {f9, f10} to change" ).c_str(), 10, 390, 20, DARKGRAY);
        }
        if (IsKeyDown(KEY_F7)) {
            updater.acc_min -= 1.0f;
        }
        if (IsKeyDown(KEY_F8)) {
            updater.acc_min += 1.0f;
        }
        if (IsKeyDown(KEY_F9)) {
            updater.acc_max -= 1.0f;
        }
        if (IsKeyDown(KEY_F10)) {
            updater.acc_max += 1.0f;
        }

    }

};