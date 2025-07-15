//// with the portion
// Copyright (C) 2018 Christoph Sommer <sommer@ccs-labs.org>
//
// Documentation for these modules is at http://veins.car2x.org/
//
// SPDX-License-Identifier: GPL-2.0-or-later
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//

#include "veins_inet/VeinsInetSampleApplication.h"

#include "inet/common/ModuleAccess.h"
#include "inet/common/packet/Packet.h"
#include "inet/common/TagBase_m.h"
#include "inet/common/TimeTag_m.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/networklayer/common/L3AddressTag_m.h"
#include "inet/transportlayer/contract/udp/UdpControlInfo_m.h"

#include "veins_inet/VeinsInetSampleMessage_m.h"

#include "inet/common/geometry/common/Coord.h"

#include <fstream>

#include "veins/modules/mobility/traci/TraCIMobility.h"
#include <cstdlib>  // For system()
#include <string>   // For std::to_string
#include <set> // Add this for the std::set
#include <random> // For random node selection

#include <cmath>     // sqrt
#include <cstring>   // memset, memcpy
#include <errno.h>   // errno

#include "inet/common/IntrusivePtr.h"

using namespace inet;

#pragma pack(push, 1)  // Ensure consistent packing
struct MessageData {
    char sender[100];
    char receiverId[100];
    double posx;
    double posy;
    double spdx;
    double spdy;
    double aclx;
    double acly;
    double hedx;
    double hedy;
    double sendTime;
    char attackType[50];
    int label;
};
#pragma pack(pop)

//____________________________________________ Static variables ________________________________ //
static std::set<std::tuple<int, int, double>> maliciousMessages; // Track malicious messages
static double maliciousMessageExpiryTime = 1.0; // Expire malicious entries after 3 seconds
//static std::vector<int> allowedNodes = {10, 12, 4, 9, 2, 6, 18, 3, 0, 1, 19, 5}; // normal message senders
//static std::vector<int> attackerNodes = {0, 3, 4, 5, 10, 18, 1, 15};

//static std::vector<int> attackerNodes = {16, 17, 4, 3, 9, 0, 15, 2, 1, 19, 11, 5};
//static std::vector<int> allowedNodes = {1, 5, 9, 15, 0, 17, 2, 7, 11, 13, 3, 8}; // eligible nodes to send messages
//static std::vector<int> allowedNodes = {0,17,16,2,9,5,19,11,4,3,15}; // eligible nodes to send messages
static std::map<std::pair<int, int>, simtime_t> lastMessageTimes; // Track last message time for each sender-receiver pair

//_______ orlando ___________//
//static std::vector<int> allowedNodes = {40, 83, 60, 74, 15, 86, 84, 48, 33, 2, 61, 92, 96, 29, 80, 70, 9, 31, 14, 64, 89, 7, 12, 71, 32, 50, 81, 30, 78, 94, 73, 34, 72, 25, 35, 93, 22, 23, 63, 5, 36, 95, 69, 24, 19, 66, 27, 56, 51, 62}; // normal message senders
//static std::vector<int> attackerNodes = {9, 63, 23, 95, 12, 24, 96, 61, 40, 31, 25, 74, 94, 50, 73, 66, 72, 69, 32, 33, 34, 86, 48, 36, 2, 71, 92, 93, 78, 5, 29, 83, 84, 15, 70, 22, 62, 30, 14, 27, 56, 81, 51, 7, 64};


//_______ casa ___________//
//static std::vector<int> allowedNodes = {26, 16, 57, 17, 41, 11, 30, 3, 58, 31, 34, 33, 44, 38, 19, 47, 6, 10, 42, 0, 18, 35, 15, 56, 37, 24, 50, 1, 25, 55, 14, 21, 59, 12, 8, 46, 28, 2, 5, 27}; // normal message senders
//static std::vector<int> attackerNodes = {17, 30, 12, 31, 14, 18, 28, 33, 6, 24, 15, 0, 16, 56, 21, 37, 26, 58, 19, 38, 11, 25, 10, 57, 34, 59, 42, 47, 2, 1, 44, 55, 50, 8, 46};

//_______ origin 20 vehicles ___________//
//static std::vector<int> allowedNodes = {11, 14, 9, 8, 17, 16, 10, 0, 13, 5, 19, 12, 7, 4, 1, 6}; // normal message senders
//static std::vector<int> attackerNodes = {6, 13, 10, 9, 4, 11, 12, 0, 5, 8};




//_______________________________________ percentages ___________________//
//_____ 20 vehicles _____//
//static std::vector<int> allowedNodes = {2, 12, 8, 0, 18, 7, 1, 16, 19, 5, 15, 13, 4, 3, 11, 9, 6, 17, 10, 14};
//static std::vector<int> attackerNodes = {2, 11}; // 10%
//static std::vector<int> attackerNodes = {2, 12, 11, 15}; // 20%
//static std::vector<int> attackerNodes = {2, 12, 11, 18, 15, 16}; // 30%
//static std::vector<int> attackerNodes = {2, 10, 11, 18, 15, 16, 14, 8, 17, 9}; // 50%


//_______________________________________ percentages ___________________//
//_____ 50 vehicles _____//
static std::vector<int> allowedNodes = {35, 1, 33, 38, 15, 0, 28, 37, 39, 14, 9, 7, 36, 48, 10, 40, 34, 46, 29, 12, 21, 11, 30, 24, 27, 6, 42, 2, 22, 20, 25, 31, 45, 32, 4, 23, 26, 3, 5, 17, 44, 41, 43, 47, 13, 49, 8, 19, 18, 16};
//static std::vector<int> attackerNodes = {17, 3, 18, 10, 6}; // 10%
//static std::vector<int> attackerNodes = {17, 3, 18, 10, 6, 9, 29, 22, 28, 14}; // 20%
//static std::vector<int> attackerNodes = {17, 3, 18, 10, 6, 9, 29, 22, 28, 14, 23, 33, 1, 30, 25}; // 30%
static std::vector<int> attackerNodes = {17, 3, 18, 10, 6, 9, 29, 22, 28, 14, 23, 33, 1, 30, 25, 27, 39, 20, 38, 31, 42, 24, 17, 19, 8}; // 50%

//_______________________________________ percentages ___________________//
//_____ 100 vehicles _____//
//static std::vector<int> allowedNodes = {41, 89, 29, 35, 55, 6, 23, 0, 77, 1, 94, 34, 87, 10, 16, 84, 57, 53, 8, 68, 70, 62, 31, 37, 96, 3, 49, 26, 83, 38, 46, 28, 12, 39, 45, 27, 75, 64, 63, 88, 86, 99, 80, 13, 91, 42, 11, 30, 4, 50, 71, 44, 43, 24, 73, 92, 78, 15, 48, 72, 17, 20, 85, 18, 32, 47, 14, 33, 93, 9, 79, 76, 21, 51, 19, 67, 5, 60, 90, 25, 58, 7, 59, 82, 74, 66, 52, 69, 2, 95, 40, 98, 65, 61, 81, 56, 97, 54, 36, 22};
//static std::vector<int> attackerNodes = {2, 15, 20, 11, 23, 7, 40, 31, 18, 16}; // 10%
//static std::vector<int> attackerNodes = {20, 2, 15, 11, 22, 33, 10, 7, 40, 31, 20, 14, 30, 39, 24, 21, 43, 34, 32, 51}; // 20%
//static std::vector<int> attackerNodes = {1, 49, 40, 57, 2, 31, 41, 15, 25, 7, 55, 18, 10, 32, 54, 28, 43, 70, 23, 51, 10, 14, 59, 95, 50, 80, 34, 69, 11, 35}; // 30%
//static std::vector<int> attackerNodes = {12, 91, 49, 55, 47, 40, 77, 41, 66, 51, 16, 54, 28, 25, 2, 9, 33, 8, 82, 21, 44, 39, 30, 6, 45, 87, 86, 96, 60, 70, 11, 31, 93, 75, 95, 62, 48, 68, 29, 36, 46, 14, 43, 63, 53, 24, 23, 50, 81, 59}; // 50%

Define_Module(VeinsInetSampleApplication);

//___________________________________ Helper function for random seed generation _____________________________//
uint32_t VeinsInetSampleApplication::getRandomSeed() {
    return static_cast<uint32_t>(
        simTime().raw() * 1000000 +
        getParentModule()->getId() +
        std::hash<std::string>{}(std::to_string(std::time(nullptr)))
    );
}

// ___________________________________ Constructor/Destructor ________________________________//
VeinsInetSampleApplication::VeinsInetSampleApplication() {}
VeinsInetSampleApplication::~VeinsInetSampleApplication() {}

//____________________________________ DoS Attack Implementation _____________________________//
void VeinsInetSampleApplication::simulateDoSDisruptive() {
    int myIndex = getParentModule()->getIndex();
    const int numAttackers = 10;
    static const double PATTERN_CHANGE_INTERVAL = 5.0;

    // Create attacker selection with new random seed
    std::vector<int> currentNodes = attackerNodes;
    std::mt19937 gen(getRandomSeed());
    std::shuffle(currentNodes.begin(), currentNodes.end(), gen);

    std::vector<int> currentAttackers(currentNodes.begin(),
        currentNodes.begin() + std::min(numAttackers, (int)currentNodes.size()));

    if (std::find(currentAttackers.begin(), currentAttackers.end(), myIndex) == currentAttackers.end()) {
        return;
    }

    // Create malicious message
    auto payload = makeShared<VeinsInetSampleMessage>();
    payload->setChunkLength(B(100));
    payload->setSenderId(getParentModule()->getFullName());
    payload->setMalicious(true);
    payload->setAttackType("DoS_Disruptive");

    // Generate malicious values
    auto position = mobility->getCurrentPosition();
    double timePhase = std::fmod(simTime().dbl(), PATTERN_CHANGE_INTERVAL) / PATTERN_CHANGE_INTERVAL;

    double baseSpeed = 30.0;
    double speedVariation = 20.0;
    double maliciousSpeedMagnitude = baseSpeed + speedVariation * std::sin(2 * M_PI * timePhase);

    double angle = 2 * M_PI * timePhase;
    double maliciousSpeedX = maliciousSpeedMagnitude * std::cos(angle);
    double maliciousSpeedY = maliciousSpeedMagnitude * std::sin(angle);

    // Set payload values
    payload->setPosx(position.x);
    payload->setPosy(position.y);
    payload->setSpdx(maliciousSpeedX);
    payload->setSpdy(maliciousSpeedY);

    double maxAccel = 8.0;
    payload->setAclx(maxAccel * std::cos(angle + M_PI/4));
    payload->setAcly(maxAccel * std::sin(angle + M_PI/4));

    // Set heading
    double magnitude = sqrt(maliciousSpeedX * maliciousSpeedX + maliciousSpeedY * maliciousSpeedY);
    if (magnitude > 0) {
        payload->setHedx(maliciousSpeedX / magnitude);
        payload->setHedy(maliciousSpeedY / magnitude);
    } else {
        payload->setHedx(0);
        payload->setHedy(0);
    }

    sendToNearbyNodes(payload, 6);
}


//_______________________________ Random Speed Attack Implementation ___________________________//
void VeinsInetSampleApplication::simulateRandomSpeed() {
    int myIndex = getParentModule()->getIndex();
    const int numAttackers = 12;

    // Select attackers
    std::vector<int> currentNodes = attackerNodes;
    std::mt19937 gen(getRandomSeed());
    std::shuffle(currentNodes.begin(), currentNodes.end(), gen);

    std::vector<int> currentAttackers(currentNodes.begin(),
        currentNodes.begin() + std::min(numAttackers, (int)currentNodes.size()));

    if (std::find(currentAttackers.begin(), currentAttackers.end(), myIndex) == currentAttackers.end()) {
        return;
    }

    auto payload = makeShared<VeinsInetSampleMessage>();
    payload->setChunkLength(B(100));
    payload->setSenderId(getParentModule()->getFullName());
    payload->setMalicious(true);
    payload->setAttackType("Random_speed");

    auto position = mobility->getCurrentPosition();
    auto speed = mobility->getCurrentVelocity();

    // Generate random speed variations
    std::normal_distribution<> speedDist(0, 15.0/3);
    double maliciousSpeedX = speed.x + speedDist(gen);
    double maliciousSpeedY = speed.y + speedDist(gen);

    payload->setPosx(position.x);
    payload->setPosy(position.y);
    payload->setSpdx(maliciousSpeedX);
    payload->setSpdy(maliciousSpeedY);

    // Generate random acceleration
    std::normal_distribution<> accelDist(0, 1.0);
    payload->setAclx(accelDist(gen));
    payload->setAcly(accelDist(gen));

    // Set heading
    double magnitude = sqrt(maliciousSpeedX * maliciousSpeedX + maliciousSpeedY * maliciousSpeedY);
    if (magnitude > 0) {
        payload->setHedx(maliciousSpeedX / magnitude);
        payload->setHedy(maliciousSpeedY / magnitude);
    } else {
        payload->setHedx(0);
        payload->setHedy(0);
    }

    sendToNearbyNodes(payload, 4);
}

//_____________________________________ Constant Position Offset Attack Implementation ____________________________//
void VeinsInetSampleApplication::simulateConstantPositionOffset() {
    int myIndex = getParentModule()->getIndex();
    const int numAttackers = 12;
    static std::map<int, Coord> positionOffsets;

    // Select attackers with new random seed
    std::vector<int> currentNodes = attackerNodes;
    std::mt19937 gen(getRandomSeed());
    std::shuffle(currentNodes.begin(), currentNodes.end(), gen);

    std::vector<int> currentAttackers(currentNodes.begin(),
        currentNodes.begin() + std::min(numAttackers, (int)currentNodes.size()));

    // Generate new offsets for current attackers
    std::normal_distribution<> offsetDist(0, 40.0);
    for (int attacker : currentAttackers) {
        if (positionOffsets.find(attacker) == positionOffsets.end()) {
            positionOffsets[attacker] = Coord(offsetDist(gen), offsetDist(gen), 0);
        }
    }

    if (std::find(currentAttackers.begin(), currentAttackers.end(), myIndex) == currentAttackers.end()) {
        return;
    }

    auto payload = makeShared<VeinsInetSampleMessage>();
    payload->setChunkLength(B(100));
    payload->setSenderId(getParentModule()->getFullName());
    payload->setMalicious(true);
    payload->setAttackType("Constant_Position_Offset");

    auto position = mobility->getCurrentPosition();
    auto speed = mobility->getCurrentVelocity();
    Coord offset = positionOffsets[myIndex];

    // Set modified position
    payload->setPosx(position.x + offset.x);
    payload->setPosy(position.y + offset.y);
    payload->setSpdx(speed.x);
    payload->setSpdy(speed.y);

    // Calculate acceleration
    double aclx = (speed.x - prevSpdx) / timeStep;
    double acly = (speed.y - prevSpdy) / timeStep;
    payload->setAclx(aclx);
    payload->setAcly(acly);

    // Update previous speeds
    prevSpdx = speed.x;
    prevSpdy = speed.y;

    // Set heading
    double magnitude = sqrt(speed.x * speed.x + speed.y * speed.y);
    if (magnitude > 0) {
        payload->setHedx(speed.x / magnitude);
        payload->setHedy(speed.y / magnitude);
    } else {
        payload->setHedx(0);
        payload->setHedy(0);
    }

    sendToNearbyNodes(payload, 4);
}

//__________________________Nearby Nodes Message Sending Implementation _____________________________//
void VeinsInetSampleApplication::sendToNearbyNodes(const inet::IntrusivePtr<VeinsInetSampleMessage>& payload, int numReceivers) {
    auto systemModule = getSimulation()->getSystemModule();
    int myIndex = getParentModule()->getIndex();
    auto position = mobility->getCurrentPosition();

    // Find valid receivers
    std::vector<cModule*> nearbyNodes;
    for (cModule::SubmoduleIterator it(systemModule); !it.end(); ++it) {
        cModule* receiverModule = *it;
        int receiverIndex = receiverModule->getIndex();

        if (receiverIndex == myIndex) continue;

        auto mobilityModule = receiverModule->getSubmodule("mobility");
        if (!mobilityModule) continue;

        auto receiverMobility = check_and_cast<veins::VeinsInetMobility*>(mobilityModule);
        double distance = sqrt(pow(position.x - receiverMobility->getCurrentPosition().x, 2) +
                             pow(position.y - receiverMobility->getCurrentPosition().y, 2));
        if (distance <= distanceThreshold) {
            nearbyNodes.push_back(receiverModule);
        }
    }

    // Randomly select and send to receivers
    // Use time-based seed for randomization
    std::mt19937 g(static_cast<unsigned int>(simTime().raw() * myIndex + nearbyNodes.size()));
    std::shuffle(nearbyNodes.begin(), nearbyNodes.end(), g);

    int actualReceivers = std::min(static_cast<int>(nearbyNodes.size()), numReceivers);
    for (int i = 0; i < actualReceivers; ++i) {
        auto receiverModule = nearbyNodes[i];
        auto l3Address = L3AddressResolver().resolve(receiverModule->getFullPath().c_str(), L3AddressResolver::ADDR_IPv4);

        auto packet = createPacket("malicious");
        packet->insertAtBack(payload->dupShared());
        packet->addTag<L3AddressReq>()->setDestAddress(l3Address);

        sendPacket(std::move(packet));

        // Log the malicious message
        int receiverIndex = receiverModule->getIndex();
        maliciousMessages.insert(std::make_tuple(myIndex, receiverIndex, simTime().dbl()));
    }
}

//______________________________________ Application Start Implementation________________________________//
bool VeinsInetSampleApplication::startApplication() {
    // Track messages sent with a precise validity time
    static std::map<std::pair<int, int>, simtime_t> sentMessages; // {sender, receiver} -> last sent time
    static bool attackTriggered = false; // Flag to track attack start

    static std::map<int, Coord> prevPositions;
    static std::map<int, Coord> prevSpeeds;
    static std::map<int, simtime_t> prevTimes;

    // Normal message callback
    auto normalMessageCallback = [this]() {
        int myIndex = getParentModule()->getIndex();

        // Get current scenario configuration
        ScenarioConfig config = getScenarioConfig();

        // Create randomized sender selection
        std::vector<int> currentNodes = allowedNodes;
        std::mt19937 gen(getRandomSeed());
        std::shuffle(currentNodes.begin(), currentNodes.end(), gen);

        if (std::find(currentNodes.begin(), currentNodes.end(), myIndex) == currentNodes.end()) {
            return;
        }

        auto systemModule = getSimulation()->getSystemModule();
        simtime_t currentTime = simTime();

        // Clean expired messages
        for (auto it = maliciousMessages.begin(); it != maliciousMessages.end();) {
            if (currentTime - std::get<2>(*it) > SimTime(maliciousMessageExpiryTime, SIMTIME_S)) {
                it = maliciousMessages.erase(it);
            } else {
                ++it;
            }
        }

        // Get current state
        auto position = mobility->getCurrentPosition();
        auto speed = mobility->getCurrentVelocity();

        // Calculate acceleration with smoothing
        double aclx = 0.0;
        double acly = 0.0;

        if (prevSpeeds.find(myIndex) != prevSpeeds.end()) {
            double dt = (currentTime - prevTimes[myIndex]).dbl();
            if (dt > 0) {
                // Calculate instantaneous acceleration
                double inst_aclx = (speed.x - prevSpeeds[myIndex].x) / dt;
                double inst_acly = (speed.y - prevSpeeds[myIndex].y) / dt;

                // Apply smoothing
                const double alpha = 0.3;
                aclx = alpha * inst_aclx + (1 - alpha) * prevSpeeds[myIndex].x;
                acly = alpha * inst_acly + (1 - alpha) * prevSpeeds[myIndex].y;

                // Limit acceleration
                const double MAX_ACCEL = 5.0;
                aclx = std::max(std::min(aclx, MAX_ACCEL), -MAX_ACCEL);
                acly = std::max(std::min(acly, MAX_ACCEL), -MAX_ACCEL);
            }
        }

        // Update state tracking
        prevPositions[myIndex] = position;
        prevSpeeds[myIndex] = speed;
        prevTimes[myIndex] = currentTime;

        // Process receivers
        std::vector<cModule*> potentialReceivers;
        for (cModule::SubmoduleIterator it(systemModule); !it.end(); ++it) {
            cModule* receiverModule = *it;
            int receiverIndex = receiverModule->getIndex();

            // Skip invalid receivers
            if (receiverIndex == myIndex) continue;

            // Skip recent malicious message pairs
            if (std::any_of(maliciousMessages.begin(), maliciousMessages.end(),
                [myIndex, receiverIndex](const std::tuple<int, int, double>& entry) {
                    return std::get<0>(entry) == myIndex &&
                           std::get<1>(entry) == receiverIndex;
                })) {
                continue;
            }

            auto mobilityModule = receiverModule->getSubmodule("mobility");
            if (!mobilityModule) continue;

            // Check distance
            auto receiverMobility = check_and_cast<veins::VeinsInetMobility*>(mobilityModule);
            auto receiverPosition = receiverMobility->getCurrentPosition();
            double distance = sqrt(pow(position.x - receiverPosition.x, 2) +
                                 pow(position.y - receiverPosition.y, 2));

            if (distance <= distanceThreshold) {
                potentialReceivers.push_back(receiverModule);
            }
        }

        // Randomize receiver order
        std::shuffle(potentialReceivers.begin(), potentialReceivers.end(), gen);

        // Process each receiver
        for (auto receiverModule : potentialReceivers) {
            int receiverIndex = receiverModule->getIndex();

            // Check message deduplication
            auto messageKey = std::make_pair(myIndex, receiverIndex);
            if (sentMessages.find(messageKey) != sentMessages.end() &&
                currentTime - sentMessages[messageKey] < SimTime(3.0, SIMTIME_S)) {
                continue;
            }

            // Update sent time
            sentMessages[messageKey] = currentTime;

            // Create and send message
            auto payload = makeShared<VeinsInetSampleMessage>();
            payload->setChunkLength(B(100));
            payload->setSenderId(getParentModule()->getFullName());

            // Set position and speed
            payload->setPosx(position.x);
            payload->setPosy(position.y);
            payload->setSpdx(speed.x);
            payload->setSpdy(speed.y);
            payload->setAclx(aclx);
            payload->setAcly(acly);

            // Calculate heading
            double speed_magnitude = sqrt(speed.x * speed.x + speed.y * speed.y);
            double headx = 0.0;
            double heady = 0.0;

            if (speed_magnitude > 0.1) {
                headx = speed.x / speed_magnitude;
                heady = speed.y / speed_magnitude;
            } else if (prevSpeeds.find(myIndex) != prevSpeeds.end()) {
                double prev_magnitude = sqrt(
                    prevSpeeds[myIndex].x * prevSpeeds[myIndex].x +
                    prevSpeeds[myIndex].y * prevSpeeds[myIndex].y
                );
                if (prev_magnitude > 0.1) {
                    headx = prevSpeeds[myIndex].x / prev_magnitude;
                    heady = prevSpeeds[myIndex].y / prev_magnitude;
                }
            }

            payload->setHedx(headx);
            payload->setHedy(heady);

            // Send packet
            auto packet = createPacket("normal");
            packet->insertAtBack(payload);
            auto l3Address = L3AddressResolver().resolve(receiverModule->getFullPath().c_str());
            packet->addTag<L3AddressReq>()->setDestAddress(l3Address);

            sendPacket(std::move(packet));
        }
    };

    // Schedule normal messages
    //timerManager.create(veins::TimerSpecification(normalMessageCallback).interval(SimTime(timeStep, SIMTIME_S)));

    // Multiple Random Speed Attack windows
    //timerManager.create(veins::TimerSpecification([this]() {simulateRandomSpeed();}).oneshotIn(SimTime(60, SIMTIME_S)));
    //timerManager.create(veins::TimerSpecification([this]() {simulateRandomSpeed();}).oneshotIn(SimTime(40, SIMTIME_S)));
    //timerManager.create(veins::TimerSpecification([this]() {simulateRandomSpeed();}).oneshotIn(SimTime(65, SIMTIME_S)));

    //timerManager.create(veins::TimerSpecification([this]() {simulateRandomSpeed();}).interval(SimTime(5, SIMTIME_S)));

    //timerManager.create(veins::TimerSpecification([this]() {simulateDoSDisruptive();}).oneshotIn(SimTime(30, SIMTIME_S)));
    //timerManager.create(veins::TimerSpecification([this]() {simulateDoSDisruptive();}).oneshotIn(SimTime(45, SIMTIME_S)));
    //timerManager.create(veins::TimerSpecification([this]() {simulateDoSDisruptive();}).oneshotIn(SimTime(70, SIMTIME_S)));
    //timerManager.create(veins::TimerSpecification([this]() {simulateDoSDisruptive();}).oneshotIn(SimTime(90, SIMTIME_S)));

    //timerManager.create(veins::TimerSpecification([this]() {simulateDoSDisruptive();}).interval(SimTime(5, SIMTIME_S)));

    //timerManager.create(veins::TimerSpecification([this]() {simulateConstantPositionOffset();}).oneshotIn(SimTime(35, SIMTIME_S)));
    //timerManager.create(veins::TimerSpecification([this]() {simulateConstantPositionOffset();}).oneshotIn(SimTime(55, SIMTIME_S)));
    //timerManager.create(veins::TimerSpecification([this]() {simulateConstantPositionOffset();}).oneshotIn(SimTime(80, SIMTIME_S)));

    //timerManager.create(veins::TimerSpecification([this]() {simulateConstantPositionOffset();}).interval(SimTime(5, SIMTIME_S)));

    //_______________________________

    // Schedule messages with randomized timing
    //double baseOffset = static_cast<double>(getParentModule()->getIndex() % 5);

    // Normal messages
    //timerManager.create(veins::TimerSpecification(normalMessageCallback).interval(SimTime(timeStep, SIMTIME_S)));

    // Attack messages with varying intervals
    //timerManager.create(veins::TimerSpecification([this]() {if (simTime() >= 10) { double randomChance = static_cast<double>(rand()) / RAND_MAX; if (randomChance < 0.5) {  simulateDoSDisruptive();}}}).interval(SimTime(3.0, SIMTIME_S)));

    // Random Speed Attack with different timing
    //timerManager.create(veins::TimerSpecification([this]() { if (simTime() >= 15) { double randomChance = static_cast<double>(rand()) / RAND_MAX; if (randomChance < 0.7) { simulateRandomSpeed();}}}).interval(SimTime(4.0, SIMTIME_S)));

    // Position Offset Attack with its own timing
    //timerManager.create(veins::TimerSpecification([this]() {if (simTime() >= 20) { double randomChance = static_cast<double>(rand()) / RAND_MAX; if (randomChance < 0.6) { simulateConstantPositionOffset();}}}).interval(SimTime(5.0, SIMTIME_S)));

    //____________________________________

    // Schedule attack windows with clear time slots
    //timerManager.create(veins::TimerSpecification([this]() {
        // Time windows for different attacks
      //  double currentTime = simTime().dbl();

        //if (currentTime >= 10 && currentTime <= 20) {
            // DoS attack window
          //  simulateDoSDisruptive();  // 60% malicious messages
        //}
        //else if (currentTime > 30 && currentTime <= 45) {
            // Random Speed attack window
          //  simulateRandomSpeed();    // 40% malicious messages
        //}
        //else if (currentTime > 50 && currentTime <= 65) {
            // Position Offset attack window
          //  simulateConstantPositionOffset();  // 30% malicious messages
        //}
    //}).interval(SimTime(3.0, SIMTIME_S)));  // Check every 3 seconds

    //________________________________________________

    // Schedule regular messages with scenario awareness
    timerManager.create(veins::TimerSpecification([this, normalMessageCallback]() {
        rotateScenario();
        ScenarioConfig config = getScenarioConfig();

        // Determine message type based on probabilities
        double random = static_cast<double>(rand()) / RAND_MAX;
        double totalAttackProb = config.dosProb + config.randomSpeedProb + config.positionProb;

        if (simTime() >= 10) {  // Start attacks after 10s
            if (random < config.dosProb) {
                simulateDoSDisruptive();
            }
            else if (random < (config.dosProb + config.randomSpeedProb)) {
                simulateRandomSpeed();
            }
            else if (random < totalAttackProb) {
                simulateConstantPositionOffset();
            }
            else {
                normalMessageCallback();
            }
        } else {
            normalMessageCallback();
        }
    }).interval(SimTime(timeStep, SIMTIME_S)));

    return true;
}


VeinsInetSampleApplication::ScenarioConfig VeinsInetSampleApplication::getScenarioConfig() {
    ScenarioConfig config;
    switch(currentScenario) {
        case SPARSE_TRAFFIC:
            config.dosProb = 0.10;         // Reduced DoS probability 0.15
            config.randomSpeedProb = 0.10;  // Reduced Random Speed 0.15
            config.positionProb = 0.25;     // Increased Position Offset 0.30
            config.messageInterval = 3;
            config.numAttackers = 10;       // Increased attackers
            break;

        case DENSE_TRAFFIC:
            config.dosProb = 0.15; // 0.20
            config.randomSpeedProb = 0.10; //0.15
            config.positionProb = 0.20;     // Maintained high 0.25
            config.messageInterval = 2;
            config.numAttackers = 12;
            break;

        case NORMAL_TRAFFIC:
            config.dosProb = 0.10; //0.15
            config.randomSpeedProb = 0.10; //0.15
            config.positionProb = 0.25;     // Increased 0.30
            config.messageInterval = 2;
            config.numAttackers = 8;
            break;
    }
    return config;
}

void VeinsInetSampleApplication::rotateScenario() {
    double currentTime = simTime().dbl();
    double scenarioDuration = 45;  // Shorter duration for each scenario

    if (currentTime - lastScenarioChange >= scenarioDuration) {
        currentScenario = static_cast<ScenarioType>((static_cast<int>(currentScenario) + 1) % 3);
        lastScenarioChange = currentTime;
        EV_INFO << "Rotating to scenario: " << currentScenario
                << " at time " << currentTime
                << " of " << scenarioDuration << "s duration" << endl;
    }
}


bool VeinsInetSampleApplication::isConnectionBlocked(const std::string& sender, const std::string& receiver) const {
    std::string connection = sender + "->" + receiver;
    return blockedConnections.find(connection) != blockedConnections.end();
}

void VeinsInetSampleApplication::updateBlockedConnections(const std::string& sender, const std::string& receiver, bool should_block) {
    std::string connection = sender + "->" + receiver;
    if (should_block) {
        blockedConnections.insert(connection);
        EV_INFO << "Blocking connection: " << connection << endl;
    } else {
        blockedConnections.erase(connection);
        EV_INFO << "Unblocking connection: " << connection << endl;
    }
}

bool VeinsInetSampleApplication::shouldProcessMessage(const std::string& sender, const std::string& receiver) const {
    return !isConnectionBlocked(sender, receiver);
}

// ___________________________________ get the message _________________________________ //
void VeinsInetSampleApplication::processPacket(std::shared_ptr<inet::Packet> pk) {
    auto payload = pk->peekAtFront<VeinsInetSampleMessage>();

    std::string senderId = payload->getSenderId();
    std::string receiverId = getParentModule()->getFullName();

    if (senderId == receiverId) {
        return;
    }

    // Check if connection is blocked before processing
    if (!shouldProcessMessage(senderId, receiverId)) {
        EV_INFO << "Message dropped - connection is blocked by GRL" << endl;
        return;
    }

    // Collect all features
    MessageData msgData = {};

    // Fill the structure
    strncpy(msgData.sender, payload->getSenderId(), 99);
    strncpy(msgData.receiverId, getParentModule()->getFullName(), 99);
    msgData.posx = payload->getPosx();
    msgData.posy = payload->getPosy();
    msgData.spdx = payload->getSpdx();
    msgData.spdy = payload->getSpdy();
    msgData.aclx = payload->getAclx();
    msgData.acly = payload->getAcly();
    msgData.hedx = payload->getHedx();
    msgData.hedy = payload->getHedy();
    msgData.sendTime = simTime().dbl();
    msgData.label = payload->getMalicious() ? 1 : 0;
    strncpy(msgData.attackType, payload->getMalicious() ? payload->getAttackType() : "normal_behavior", 49);

    // Add this before sending data
    EV_INFO << "Sending data: size=" << sizeof(msgData)
            << ", senderId=" << msgData.sender
            << ", receiverId=" << msgData.receiverId << endl;


    // Create socket
    SOCKET sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) {
        EV_ERROR << "Socket creation error: " << WSAGetLastError() << endl;
        return;
    }

    // Setup server address
    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PYTHON_SERVER_PORT);

    unsigned long addr = inet_addr(PYTHON_SERVER_IP);
    if (addr == INADDR_NONE) {
        EV_ERROR << "Invalid IP address" << endl;
        closesocket(sock);
        return;
    }
    serv_addr.sin_addr.s_addr = addr;

    // Connect to Python server
    if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) == SOCKET_ERROR) {
        EV_ERROR << "Connection Failed: " << WSAGetLastError() << endl;
        closesocket(sock);
        return;
    }

    // Send all data
    if (::send(sock, (char*)&msgData, sizeof(msgData), 0) == SOCKET_ERROR) {
        EV_ERROR << "Send failed: " << WSAGetLastError() << endl;
        closesocket(sock);
        return;
    }

    // Receive both prediction and GRL decision
    int response[2];  // [prediction, should_prune]
    if (recv(sock, (char*)&response, sizeof(response), 0) == SOCKET_ERROR) {
        EV_ERROR << "Receive failed: " << WSAGetLastError() << endl;
        closesocket(sock);
        return;
    }

    int prediction = response[0];
    int should_prune = response[1];

    // Update blocked connections based on GRL decision
    updateBlockedConnections(senderId, receiverId, should_prune == 1);

    // Log results
    EV_INFO << "Message from " << senderId << " to " << receiverId
            << " classified as: " << (prediction == 1 ? "Malicious" : "Normal")
            << ", GRL decision: " << (should_prune == 1 ? "Prune" : "Maintain") << endl;

    closesocket(sock);


    // Log to file
        std::ofstream logFile;
        bool isNewFile = !std::ifstream("C:/Users/Latifa/src/vanetTuto/simulations/veins_inet_openStreetMap/AD/data/temp_sim.csv");

        logFile.open("C:/Users/Latifa/src/vanetTuto/simulations/veins_inet_openStreetMap/AD/data/temp_sim.csv", std::ios::app);
        if (isNewFile) {
            logFile << "sendTime,SenderID,ReceiverID,PosX,PosY,SpdX,SpdY,AclX,AclY,HedX,HedY,AttackType,Status,Prediction,GRLAction\n";
        }

        logFile << msgData.sendTime << ","
                << msgData.sender << ","
                << msgData.receiverId << ","
                << msgData.posx << ","
                << msgData.posy << ","
                << msgData.spdx << ","
                << msgData.spdy << ","
                << msgData.aclx << ","
                << msgData.acly << ","
                << msgData.hedx << ","
                << msgData.hedy << ","
                << msgData.attackType << ","
                << msgData.label << ","
                << prediction << ","
                << should_prune << "\n";
        logFile.close();

    // Call Python script to log messgs
    std::string command = "python3 C:/Users/Latifa/src/vanetTuto/simulations/veins_inet_openstreetMap/python_openstreetMap.py " +
            std::string(msgData.sender) + " " +
            std::string(msgData.receiverId) + " " +
            std::to_string(msgData.posx) + " " +
            std::to_string(msgData.posy) + " " +
            std::to_string(msgData.spdx) + " " +
            std::to_string(msgData.spdy) + " " +
            std::to_string(msgData.aclx) + " " +
            std::to_string(msgData.acly) + " " +
            std::to_string(msgData.hedx) + " " +
            std::to_string(msgData.hedy) + " " +
            std::to_string(msgData.label) + " " +
            std::string(msgData.attackType) + " " +
            std::to_string(msgData.sendTime);

      system(command.c_str());

    EV_INFO << "Message from " << senderId << " to " << receiverId
            << " classified as: " << (prediction == 1 ? "Malicious" : "Normal")
            << ", GRL decision: " << (should_prune == 1 ? "Prune" : "Keep") << endl;
}


//______________________ Initialization and cleanup ___________________________________//
void VeinsInetSampleApplication::initialize(int stage) {
    veins::VeinsInetApplicationBase::initialize(stage);

    // Initialize random seed
    srand(static_cast<unsigned int>(std::time(nullptr)) + getParentModule()->getIndex());

    // Initialize Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        EV_ERROR << "WSAStartup failed" << endl;
        return;
    }
}

void VeinsInetSampleApplication::finish() {
    WSACleanup();
}

bool VeinsInetSampleApplication::stopApplication() {}
