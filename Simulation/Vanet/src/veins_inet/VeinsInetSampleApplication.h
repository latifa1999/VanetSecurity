#pragma once

#include "veins_inet/veins_inet.h"

#include "veins_inet/VeinsInetApplicationBase.h"

#include "veins/veins.h"
#include "veins/modules/application/ieee80211p/DemoBaseApplLayer.h"
#include "veins/modules/mobility/traci/TraCIMobility.h"
#include "veins/modules/mobility/traci/TraCICommandInterface.h"
#include "veins_inet/VeinsInetMobility.h"
#include "inet/common/packet/Packet.h"
#include "veins_inet/veins_inet.h"
#include <winsock2.h>
#include <ws2tcpip.h>

#include "inet/common/geometry/common/Coord.h"
#include "veins_inet/VeinsInetSampleMessage_m.h"
#include "inet/common/IntrusivePtr.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "veins/base/utils/FindModule.h"

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")


class VEINS_INET_API VeinsInetSampleApplication : public veins::VeinsInetApplicationBase {
protected:
    // Define ScenarioType enum inside the class
    enum ScenarioType {
        SPARSE_TRAFFIC = 0,
        NORMAL_TRAFFIC = 1,
        DENSE_TRAFFIC = 2
    };

    // Define ScenarioConfig struct inside the class
    struct ScenarioConfig {
        double dosProb;        // DoS attack probability
        double randomSpeedProb;// Random speed attack probability
        double positionProb;   // Position offset attack probability
        int messageInterval;   // Base message interval
        int numAttackers;     // Number of attackers for each type

        // Constructor with default values
        ScenarioConfig() :
            dosProb(0.25),
            randomSpeedProb(0.2),
            positionProb(0.2),
            messageInterval(3),
            numAttackers(10)
        {}
    };

    // Scenario management members
    ScenarioType currentScenario = NORMAL_TRAFFIC;
    int scenarioDuration = 25;  // Duration of each scenario in seconds
    double lastScenarioChange = 0;

    // Add method declarations
    ScenarioConfig getScenarioConfig();
    void rotateScenario();

    // Simulation control
    void finish() override;
    virtual bool startApplication() override;
    virtual bool stopApplication() override;
    virtual void initialize(int stage) override;

    // Message handling
    virtual void processPacket(std::shared_ptr<inet::Packet> pk) override;

    // python server
    void connectToPythonServer(const char* ip, int port);

    // Scenario management methods
    void updateAttackProbabilities();

    // Attack simulations
    void simulateRandomSpeed();
    void simulateDoSDisruptive();
    void simulateConstantPositionOffset();
    void simulateConstantSpeed();

    // Helper functions
    void sendToNearbyNodes(const inet::IntrusivePtr<VeinsInetSampleMessage>& payload,  int maliciousRatio);
    uint32_t getRandomSeed();


protected:
    // GRL-related members
    std::set<std::string> blockedConnections;  // Store pruned connections

    // Helper functions for GRL
    bool isConnectionBlocked(const std::string& sender, const std::string& receiver) const;
    void updateBlockedConnections(const std::string& sender, const std::string& receiver, bool should_block);
    bool shouldProcessMessage(const std::string& sender, const std::string& receiver) const;

protected:
    // Socket related constants
    const char* PYTHON_SERVER_IP = "127.0.0.1";
    const int PYTHON_SERVER_PORT = 5000;

    // Previous velocities for calculations
    double prevSpdx = 0.0;
    double prevSpdy = 0.0;

    const double distanceThreshold = 400.0;
    const double timeStep = 1.0;

public:
    VeinsInetSampleApplication();
    ~VeinsInetSampleApplication();
};
