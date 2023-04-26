#include "../inc/InverseKinematics.hpp"


InverseKinematics::InverseKinematics(rw::models::SerialDevice::Ptr device, rw::core::Ptr<rw::models::WorkCell> wc, rw::kinematics::State state)
{
    // Set robot and workcell
    this->workCell_ = wc;
    this->robot_ = device;
    this->state_ = state;

    // Create inverse kinematics solver
    this->solver_ = rw::common::ownedPtr(new rw::invkin::ClosedFormIKSolverUR(this->robot_, this->state_));
    
    // Create collision detector
    this->collisionStrategy_ = rwlibs::proximitystrategies::ProximityStrategyYaobi::make();
    this->collisionDetector_ = rw::common::ownedPtr(new rw::proximity::CollisionDetector(this->workCell_, this->collisionStrategy_));

}

InverseKinematics::~InverseKinematics()
{
}

bool InverseKinematics::solve(rw::math::Transform3D<> WorldTTarget, double rollStart, double rollEnd, double step)
{
    // Find known kinematics frames
    rw::kinematics::Frame* frameTcp = this->workCell_->findFrame("GraspTCP");
    rw::kinematics::Frame* frameRobotBase = this->workCell_->findFrame("UR-6-85-5-A.Base");
    rw::kinematics::Frame* frameRobotTcp = this->workCell_->findFrame("UR-6-85-5-A.TCP");

    // Find known transforms
    rw::math::Transform3D<> frameWorldTBase = rw::kinematics::Kinematics::worldTframe(frameRobotBase, this->state_);
    rw::math::Transform3D<> frameTcpTRobotTcp = rw::kinematics::Kinematics::frameTframe(frameTcp, frameRobotTcp, this->state_);    
    rw::math::Transform3D<> frameBaseTGoal = rw::math::Transform3Dd::invMult(frameWorldTBase, WorldTTarget);

    if (rollEnd == invalid)
        rollEnd = rollStart;

    auto state_copy = this->state_.clone(); 
    double timeColFree = 0.0;
    bool solutionFound = false;
    double timeCol = 0.0;

    // Solve inverse kinematics
    for (double rollAngle = rollStart; rollAngle <= rollEnd; rollAngle += step)
    {   
        rw::math::Transform3D<> graspTcpTroll = rw::math::Transform3D<>(
            rw::math::Vector3D<double>(0,0,0), 
            rw::math::RPY<double>(rollAngle, 0, 0).toRotation3D()
        );

        rw::math::Transform3D<> targetAt = (frameBaseTGoal 
            * graspTcpTroll // rotate TCP to find multiple solutions
            * frameTcpTRobotTcp // offset TCP to be at UR5 TCP from tool TCP
        );

        std::vector<rw::math::Q> sub_solutions = this->solver_->solve(targetAt, this->state_);
        for (auto q : sub_solutions)
        {
            this->robot_->setQ(q, state_copy);
            if (!this->collisionDetector_->inCollision(state_copy))
            {
                solutionFound = true;
                this->collisionFreeSolutions_.push_back(q);
                this->collisionFreeStates_.push_back(rw::trajectory::TimedState(timeColFree, state_copy));
                timeColFree += 0.1;
            }
            else
            {
                this->collisionStates_.push_back(rw::trajectory::TimedState(timeCol, state_copy));
                timeCol += 0.1;
            }
        }
    }
    return solutionFound;
}

std::vector<rw::math::Q> InverseKinematics::getSolutions()
{
    return this->collisionFreeSolutions_;
}

void InverseKinematics::getSolutions(std::vector<rw::math::Q>& solutions)
{
    solutions = this->collisionFreeSolutions_;
}



rw::trajectory::TimedStatePath InverseKinematics::getReplay()
{
    return this->collisionFreeStates_;
}

void InverseKinematics::getReplay(rw::trajectory::TimedStatePath& replay)
{
    replay = this->collisionFreeStates_;
}


